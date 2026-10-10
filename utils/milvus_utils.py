# utils/milvus_utils.py

import os
import time

from tool.logger import logger
from pymilvus import MilvusClient, AnnSearchRequest, WeightedRanker, DataType

from config.index_config import index_config
from config.milvus_config import milvus_config


_milvus_client = None
# 上次连接失败的时间戳。连一次要跑完 urllib3 的 5 次重试（约 30s），Milvus 挂掉时
# 如果每个请求、每次指标抓取都去重连，接口会被自己的重试拖死，所以失败后短路一小段时间
_milvus_failed_at = 0.0
_RETRY_INTERVAL_SECONDS = 10.0


def get_milvus_client():
    """
    获取全局单例的MilvusClient对象
    MilvusClient 构造时会立刻连一次服务，连不上直接抛异常；这里统一转成返回 None，
    让所有调用方的 `if not client` 分支真正生效，也避免 Milvus 没起来时接口返回裸堆栈
    :return: MilvusClient实例；连接失败返回 None
    """
    global _milvus_client, _milvus_failed_at
    if _milvus_client is not None:
        return _milvus_client
    if time.time() - _milvus_failed_at < _RETRY_INTERVAL_SECONDS:
        return None

    try:
        client = MilvusClient(uri=milvus_config.milvus_url)
    except Exception as e:
        # 不缓存失败结果：Milvus 恢复后过了重试间隔就能自动重连
        _milvus_failed_at = time.time()
        logger.error(f"Milvus 连接失败（{milvus_config.milvus_url}）: {e}")
        return None

    _milvus_client = client
    return _milvus_client



def escape_milvus_string(value: str) -> str:
    """
    Milvus数据库过滤表达式中字符串的安全转义函数（防止解析失败）
    作用：
        转义特殊字符（反斜杠、双引号），避免Milvus解析filter时报错
    参数：
        value: 需要转义的原始字符串
    返回：
        str: 转义后的安全字符串
    """
    # 转义反斜杠（\ → \\） 双引号（" → \"） 单引号（' → \'）
    value = value.replace("\\", "\\\\").replace('"', '\\"').replace("'", "\\'")
    return value


def build_dense_params(strategy: dict = None) -> dict:
    """
    按意图策略构造稠密向量搜索参数（HNSW 的 ef 等）
    策略未指定 ef_search（0/None）时不传 params，沿用索引自身默认值
    :param strategy: 意图策略字典，取 ef_search 字段
    :return: Milvus 稠密向量搜索参数字典
    """
    dense_params = {"metric_type": "COSINE"}
    ef_search = (strategy or {}).get("ef_search")
    if ef_search:
        dense_params["params"] = {"ef": ef_search}
    return dense_params


def build_chunks_schema(client, vector_dimension: int):
    """
    构建切片集合的 Schema（导入节点与索引重建共用，避免两处字段定义漂移）
    :param client: MilvusClient实例
    :param vector_dimension: 稠密向量维度
    :return: CollectionSchema
    """
    schema = client.create_schema(auto_id=True, enable_dynamic_field=True)
    schema.add_field(field_name="chunk_id", datatype=DataType.INT64, is_primary=True, auto_id=True)
    schema.add_field(field_name="content", datatype=DataType.VARCHAR, max_length=65535)  # 切片内容
    schema.add_field(field_name="title", datatype=DataType.VARCHAR, max_length=100)  # 切片标题
    schema.add_field(field_name="parent_title", datatype=DataType.VARCHAR, max_length=100)  # 父标题
    schema.add_field(field_name="part", datatype=DataType.INT8)  # 分片编号
    schema.add_field(field_name="file_title", datatype=DataType.VARCHAR, max_length=100)  # 源文件标题
    schema.add_field(field_name="item_name", datatype=DataType.VARCHAR, max_length=100)  # 商品名称（幂等性依据）
    schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)  # 稀疏向量
    schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=vector_dimension)  # 稠密向量
    return schema


def build_chunks_index_params(client, dense_index=None, sparse_index=None):
    """
    构建切片集合的索引参数

    为什么不用 AUTOINDEX：AUTOINDEX 由 Milvus 自行决定索引类型与参数，既不可控也没法调优。
    这里显式指定 HNSW（M / efConstruction），召回与延迟可以通过参数权衡，重建时也能换参数对比。

    :param client: MilvusClient实例
    :param dense_index: 稠密向量索引规格，默认取 config.index_config
    :param sparse_index: 稀疏向量索引规格，默认取 config.index_config
    :return: IndexParams
    """
    dense_index = dense_index or index_config.dense_index
    sparse_index = sparse_index or index_config.sparse_index

    index_params = client.prepare_index_params()
    index_params.add_index(
        field_name="dense_vector",
        index_name="dense_vector_index",
        index_type=dense_index.index_type,
        metric_type=dense_index.metric_type,
        params=dense_index.index_params(),
    )
    index_params.add_index(
        field_name="sparse_vector",
        index_name="sparse_inverted_index",
        index_type=sparse_index.index_type,
        metric_type=sparse_index.metric_type,
        params=sparse_index.index_params(),
    )
    return index_params


def create_chunks_collection(client, collection_name: str, vector_dimension: int,
                             dense_index=None, sparse_index=None) -> bool:
    """
    创建切片集合（Schema + 索引一次建好）
    :param client: MilvusClient实例
    :param collection_name: 集合名（物理名或逻辑名都可以）
    :param vector_dimension: 稠密向量维度
    :param dense_index: 稠密向量索引规格，默认取配置
    :param sparse_index: 稀疏向量索引规格，默认取配置
    :return: 是否创建成功
    """
    try:
        client.create_collection(
            collection_name=collection_name,
            schema=build_chunks_schema(client, vector_dimension),
            index_params=build_chunks_index_params(client, dense_index, sparse_index),
        )
        logger.info(f"Milvus集合[{collection_name}]创建完成，维度={vector_dimension}，"
                    f"稠密索引={(dense_index or index_config.dense_index).index_type}")
        return True
    except Exception as e:
        logger.error(f"Milvus集合[{collection_name}]创建失败: {str(e)}", exc_info=True)
        return False


def get_vector_dim(client, collection_name: str) -> int:
    """
    读取集合的稠密向量维度（重建新集合时要和旧集合保持一致）
    :return: 维度；读取失败返回 0
    """
    try:
        desc = client.describe_collection(collection_name)
        for field in desc.get("fields", []):
            if field.get("name") == "dense_vector":
                return int(field.get("params", {}).get("dim", 0))
    except Exception as e:
        logger.error(f"读取集合[{collection_name}]向量维度失败: {str(e)}")
    return 0


def get_row_count(client, collection_name: str) -> int:
    """
    读取集合行数（一致性校验用；Milvus 统计有延迟，flush 后才准）
    :return: 行数；读取失败返回 -1
    """
    try:
        return int(client.get_collection_stats(collection_name).get("row_count", 0))
    except Exception as e:
        logger.error(f"读取集合[{collection_name}]行数失败: {str(e)}")
        return -1


def count_rows(client, collection_name: str) -> int:
    """
    精确统计集合行数
    get_collection_stats 的 row_count 把「已删除但还没压缩」的数据也算进去，会偏大；
    一致性校验要拿它做新旧集合对比，所以用 count(*) 走一次真实查询。
    :return: 行数；两种方式都失败返回 -1
    """
    try:
        rows = client.query(collection_name=collection_name, filter="", output_fields=["count(*)"])
        return int(rows[0]["count(*)"])
    except Exception as e:
        logger.warning(f"集合[{collection_name}]精确计数失败，回退统计值: {str(e)}")
        return get_row_count(client, collection_name)


def alias_target(client, alias: str) -> str:
    """
    查询别名当前指向哪个物理集合
    :return: 物理集合名；别名不存在返回空字符串
    """
    try:
        return (client.describe_alias(alias) or {}).get("collection", "") or ""
    except Exception:
        # 别名不存在时 Milvus 会抛错，这是正常路径（还没做过迁移）
        return ""


def create_hybrid_search_requests(dense_vector, sparse_vector, dense_params=None, sparse_params=None, expr=None,
                                  limit=5):
    """
    构建Milvus混合搜索请求对象
    分别创建稠密/稀疏向量的搜索请求，用于后续混合搜索融合
    :param dense_vector: 文本生成的稠密向量
    :param sparse_vector: 文本生成的稀疏向量
    :param dense_params: 稠密向量搜索参数，默认使用余弦相似度
    :param sparse_params: 稀疏向量搜索参数，默认使用内积相似度
    :param expr: 搜索过滤表达式，用于精准筛选数据
    :param limit: 单向量搜索返回结果数量，默认5
    :return: 搜索请求列表，包含[dense_req, sparse_req]
    """
    # 稠密向量默认搜索参数：余弦相似度（COSINE），适配BGE-M3稠密向量
    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    # 稀疏向量默认搜索参数：内积（IP），适配BGE-M3稀疏向量
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    # 构建稠密向量搜索请求，关联Milvus的dense_vector字段 近似最近邻（ANN）检索请求的核心类
    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        limit=limit
    )

    # 构建稀疏向量搜索请求，关联Milvus的sparse_vector字段
    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        limit=limit
    )

    return [dense_req, sparse_req]


def hybrid_search(client, collection_name, reqs, ranker_weights=(0.5, 0.5), norm_score=False, limit=5,
                  output_fields=None, search_params=None):
    """
    执行Milvus稠密+稀疏向量混合搜索
    基于WeightedRanker实现双向量搜索结果加权融合，提升检索准确性
    :param client: MilvusClient实例
    :param collection_name: 集合名称
    :param reqs: 搜索请求列表，固定为[dense_req, sparse_req]
    :param ranker_weights: 加权融合权重，默认(0.5,0.5)，依次对应稠密/稀疏向量
    :param norm_score: 是否归一化评分后再融合，避免评分量级差异导致权重失效
    :param limit: 混合搜索最终返回结果数量，默认5
    :param output_fields: 需要返回的字段列表，默认返回item_name
    :param search_params: 搜索参数，如ef/topk等，默认None
    :return: 混合搜索结果列表，搜索失败返回None
    """
    # client 为 None 时直接返回：否则报出来的是 'NoneType' object has no attribute 'hybrid_search'，
    # 一堆堆栈把真正的原因（Milvus 连不上）盖住了
    if client is None:
        logger.warning(f"Milvus 客户端不可用，跳过混合搜索，集合[{collection_name}]")
        return None
    try:
        # 初始化加权排名器：按权重融合稠密/稀疏向量的搜索结果
        # norm_score=True：先将两个向量评分归一化到0~1区间，再加权计算，避免一个得分特别大、另一个特别小导致权重失效。
        # 版本：V2.4
        rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)
        # 默认返回字段：文档标识字段
        if output_fields is None:
            output_fields = ["item_name"]

        # 执行混合搜索：融合稠密+稀疏向量结果，按权重重新排序
        res = client.hybrid_search(
            collection_name=collection_name,
            reqs=reqs,
            ranker=rerank,
            limit=limit,
            output_fields=output_fields,
            search_params=search_params
        )

        logger.info(f"Milvus混合搜索完成，集合[{collection_name}]共检索到{len(res[0])}条结果")
        return res
    except Exception as e:
        logger.error(f"Milvus混合搜索执行失败，集合[{collection_name}]：{str(e)}", exc_info=True)
        return None