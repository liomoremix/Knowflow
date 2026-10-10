# processor/import_processor/nodes/node_import_milvus.py
import json
import logging
from typing import Dict, Any, List

from processor.import_processor.base import BaseNode, setup_logging
from processor.import_processor.exceptions import StateFieldError, MilvusError
from processor.import_processor.state import ImportGraphState
from utils.index_registry_utils import get_write_targets, mark_physical_failed
from utils.milvus_utils import get_milvus_client, escape_milvus_string, create_chunks_collection


class NodeImportMilvus(BaseNode):
    """
    导入向量库节点：数据持久化
    """

    name = "node_import_milvus"

    name: str = "node_import_milvus"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        LangGraph核心节点：Milvus切片数据入库主流程
        执行流程（串行执行，一步一校验，保证数据一致性）：
            1. 输入校验：验证切片有效性、向量字段完整性，提取向量维度
            2. 环境准备：连接Milvus，集合不存在则自动创建Schema+索引
            3. 幂等清理：删除同file_title旧数据，避免重复存储
            4. 批量插入：预处理数据后批量入库，回填Milvus自增chunk_id
            5. 状态更新：将回填了chunk_id的切片更新回全局状态，供下游使用

        蓝绿重建期间（索引注册表里有在建版本）步骤2会返回两个写入目标，
        步骤3/4 对新旧集合同时生效：主集合失败终止导入，在建副本失败只记日志并退出双写。

        异常处理：
            任一步骤失败抛出异常，终止节点执行，保证数据不脏写

        必要参数：chunks
        更新参数：chunks字段回填chunk_id

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 步骤1：输入数据有效性校验
        chunks_json_data, vector_dimension = self._step_1_check_input(state)

        # 步骤2：Milvus客户端连接+集合准备（自动建表；蓝绿重建期间返回双写目标）
        client, write_targets = self._step_2_prepare_collection(vector_dimension)

        # 步骤3：幂等性处理 - 清理同file_title旧数据（逐个写入目标清理，失败的副本被剔除）
        write_targets = self._step_3_clean_old_data(client, chunks_json_data, write_targets)

        # 步骤4：批量插入数据+主键chunk_id回填（主集合必须成功，副本失败不阻塞）
        updated_chunks = self._step_4_insert_data(client, chunks_json_data, write_targets)

        # 步骤5：更新全局状态，将回填后的切片回传下游
        state["chunks"] = updated_chunks

        return state




    def _step_1_check_input(self, state: Dict[str, Any]) -> tuple[List[Dict[str, Any]], int]:
        """
        步骤1：输入数据有效性校验
        核心校验项：
            1. chunks非空且为列表类型
            2. 切片包含dense_vector核心字段
            3. 提取向量维度，为集合创建/索引构建提供依据
        参数：
            state: Dict[str, Any] - 流程状态对象，包含上游传入的chunks数据
        返回：
            tuple - (校验通过的切片列表, 稠密向量维度)
        异常：
            任一校验项不通过，抛出ValueError终止入库流程，避免脏数据处理

        """

        # 校验1：chunks非空
        chunks = state.get("chunks")

        if not chunks:
            raise StateFieldError(field_name="chunks", message="chunks不能为空", expected_type=list)

        if not isinstance(chunks, list):
            raise StateFieldError(field_name="chunks", message="chunks数据类型不正确", expected_type=list)

        # 校验2：切片包含dense_vector字段
        first_chunk = chunks[0]
        if 'dense_vector' not in first_chunk:
            raise StateFieldError(field_name="chunks", message="错误: 数据中缺失dense_vector字段")

        # 校验3：切片包含 sparse_vector 字段
        if 'sparse_vector' not in first_chunk:
            raise StateFieldError(field_name="chunks", message="错误: 数据中缺失sparse_vector字段")

        # 提取向量维度
        vector_dimension = len(first_chunk['dense_vector'])
        return chunks, vector_dimension




    def _step_2_prepare_collection(self, vector_dimension: int):
        """
        步骤2：Milvus客户端连接+集合准备
        核心逻辑：
            1. 获取Milvus单例客户端，验证连接有效性
            2. 从索引注册表取写入目标（单写 active，或重建期间双写 active+building）
            3. 集合不存在则自动创建（Schema+索引），存在则直接复用
        参数：
            vector_dimension: int - 稠密向量维度（步骤1提取）
        返回：
            tuple - (MilvusClient实例, 写入目标集合名列表)，列表第 0 个是主集合
        异常：
            客户端获取失败/主集合创建失败，抛出异常终止流程
        """

        # 1. 获取milvus客户端对象
        milvus_client = get_milvus_client()
        if not milvus_client:
            self.logger.error("Milvus 连接失败")
            raise MilvusError("Milvus 连接失败")

        # 2. 确定写入目标：正常只有 active 集合；蓝绿重建期间是 [active, building] 双写。
        #    拿的是物理集合名而不是 alias——写入必须落到确定的集合，否则两次都写进同一个。
        write_targets = get_write_targets()

        # 3. 集合不存在则创建
        for position, collection_name in enumerate(write_targets):
            if milvus_client.has_collection(collection_name):
                continue

            required = position == 0
            created = create_chunks_collection(milvus_client, collection_name, vector_dimension)
            if created:
                continue

            if required:
                raise MilvusError(f"Milvus 集合[{collection_name}]创建失败")
            # 在建副本建不出来：退出双写，不影响线上导入
            self._on_replica_failure(collection_name, "集合创建失败")
            write_targets = write_targets[:position]
            break

        return milvus_client, write_targets

    def _on_replica_failure(self, physical: str, error: str):
        """
        双写副本（在建的新集合）出问题的统一处理：
        把该版本标记为 failed 退出双写，线上导入照常完成，重建任务留给人工重跑。
        副本失败绝不能阻塞主集合导入——那会让一次索引重建把正常入库也拖挂。
        """
        self.logger.error(f"双写副本[{physical}]失败，已退出双写，需人工重跑重建: {error}")
        mark_physical_failed(physical, error)

    def _step_3_clean_old_data(self, client, chunks_json_data, write_targets) -> List[str]:

        """
        幂等清理
        基于每个片段的file_title进行旧数据的清理（每个写入目标都要清，否则新集合会残留旧版本数据）
        :param client: milvus客户端
        :param chunks_json_data: chunks数据
        :param write_targets: 写入目标集合名列表，第 0 个是主集合
        :return: 清理成功的写入目标列表（副本清理失败会被剔除，不能带着旧数据继续插）
        """

        # 1. 获取查询条件
        file_title = chunks_json_data[0].get("file_title")

        # 2. 逐个目标执行幂等清理，只保留清理成功的
        survivors = []
        for position, collection_name in enumerate(write_targets):
            if self._clear_chunks_by_file_title(client, collection_name, file_title, required=position == 0):
                survivors.append(collection_name)
        return survivors

    def _clear_chunks_by_file_title(self, client, collection_name, file_title, required=True) -> bool:
        """
        删除某个集合里同 file_title 的旧数据
        :param required: True 表示主集合，失败要抛异常终止导入；False 表示副本，失败只退出双写
        :return: 是否清理成功
        """
        try:
            file_title = escape_milvus_string(file_title)
            client.delete(
                collection_name=collection_name,
                filter=f"file_title=='{file_title}'")
            return True
        except Exception as e:
            self.logger.error(f"Milvus 数据删除失败[{collection_name}]: {str(e)}")
            if required:
                raise MilvusError(f"Milvus 数据删除失败: {str(e)}")
            self._on_replica_failure(collection_name, f"幂等清理失败: {str(e)}")
            return False



    def _step_4_insert_data(self, client, chunks_json_data: List[Dict[str, Any]],
                            write_targets: List[str]) -> List[Dict[str, Any]]:
        """
        步骤4：批量插入切片数据到Milvus+主键回填
        核心逻辑：
            1. 批量插入数据：提升入库效率，减少Milvus连接次数
            2. 双写：重建期间同时写入在建的新集合，保证切换时新集合不缺这批数据
            3. 回填chunk_id：只取主集合返回的自增主键（新旧集合的 chunk_id 不相同也不需要同步）
        参数：
            client - MilvusClient实例
            chunks_json_data: List[Dict[str, Any]] - 待入库的切片列表
            write_targets: List[str] - 写入目标集合名列表，第 0 个是主集合
        返回：
            List[Dict[str, Any]] - 回填了chunk_id的切片列表
        """
        # 1. 预处理数据：移除手动chunk_id，避免与Milvus自增主键冲突
        data_to_insert = []
        for item in chunks_json_data:
            item_copy = item.copy()

            # 补充 part 字段
            if "part" not in item_copy:
                item_copy["part"] = 0

            # 添加到待插入列表
            data_to_insert.append(item_copy)

        # 2. 主集合插入：失败直接抛异常终止导入
        insert_result = self._insert_into(client, write_targets[0], data_to_insert, required=True)

        # 3. 副本集合插入（蓝绿重建期间的新集合）：失败不阻塞主流程
        for collection_name in write_targets[1:]:
            self._insert_into(client, collection_name, data_to_insert, required=False)

        # 4. 主键回填：将Milvus生成的chunk_id回填到原始切片
        inserted_ids = insert_result.get('ids', [])
        if inserted_ids:
            for idx, item in enumerate(chunks_json_data):
                item['chunk_id'] = str(inserted_ids[idx])

        return chunks_json_data

    def _insert_into(self, client, collection_name: str, data_to_insert: List[Dict[str, Any]],
                     required: bool) -> Dict[str, Any]:
        """向单个集合批量插入；required=False 时失败只记日志并退出双写"""
        try:
            insert_result = client.insert(collection_name=collection_name, data=data_to_insert)
            self.logger.info(f"Milvus 插入完成[{collection_name}]: {insert_result.get('insert_count', 0)} 条")
            return insert_result
        except Exception as e:
            self.logger.error(f"Milvus 数据插入失败[{collection_name}]: {str(e)}")
            if required:
                raise MilvusError(f"Milvus 数据插入失败: {str(e)}")
            self._on_replica_failure(collection_name, f"插入失败: {str(e)}")
            return {}





if __name__ == "__main__":

    setup_logging()

    json_path = r"D:\output\hak180产品安全手册\state_vector.json"
    with open(json_path, "r", encoding="utf-8") as f:
        state_json = f.read()

    state = json.loads(state_json)

    init_state = {
        "chunks": state.get("chunks")
    }

    # 执行核心处理流程
    node_import_milvus = NodeImportMilvus()
    result = node_import_milvus(init_state)

    logging.getLogger().info(json.dumps(result, ensure_ascii=False, indent=4))