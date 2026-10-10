# processor/query_processor/nodes/node_search_embedding.py

import time

from config.milvus_config import milvus_config
from processor.query_processor.base import NodeBase, T
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.json_format_utils import serialize_json
from utils.metrics_utils import observe_node_error, observe_retrieval
from utils.milvus_utils import build_dense_params, create_hybrid_search_requests, get_milvus_client, hybrid_search
from utils.query_trace_utils import path_metric


class NodeSearchEmbedding(NodeBase):
     """
    节点功能：基于已确认主体名+改写后的用户问题，执行Milvus向量数据库混合检索
    """

     # 覆盖基类的 name 属性，标识节点名称
     name: str = "node_search_embedding"

     def process(self, state: QueryGraphState) -> QueryGraphState:
         """
         核心节点函数：基于已确认商品名+改写后的用户问题，执行Milvus向量数据库混合检索
         流程：用户问题向量化 → 构造带商品名过滤的混合搜索请求 → 执行稠密+稀疏混合检索 → 返回检索结果
         :param state: Dict - 会话状态字典，包含上游传递的核心信息，关键字段：
                       {
                           "rewritten_query": str,   # step4改写后的完整用户问题（含商品名）
                           "item_names": list[str],  # step7已确认的标准化商品名列表
                       }

         :return: Dict - 检索结果字典，仅包含embedding_chunks字段，供下游节点使用：
                  {
                      "embedding_chunks": List[Dict]  # Milvus检索结果列表，无结果则为空列表
                                                      # 每个元素为一条匹配的向量数据，含业务字段
                      "dense_path_metric": Dict       # 本路的耗时/命中/参数（监控与慢查询日志用）
                  }
         """

         # 计时放在 try 外：异常分支也要能报出「这一路花了多久后失败的」
         started = time.perf_counter()

         try:

             # 1、用户问题和已确认商品名
             query = state.get("rewritten_query")
             item_names = state.get("item_names")

             # 2、生成向量 (Dense + Sparse)：优先复用意图节点已算一次的向量，避免重复 embedding
             strategy = state.get("strategy") or {}
             dense_vec = state.get("query_dense_vec")
             sparse_vec = state.get("query_sparse_vec")
             if not dense_vec:
                 embeddings = generate_embeddings([query])
                 dense_vec = embeddings.get("dense")[0]
                 sparse_vec = embeddings.get("sparse")[0]

             # 3. 获取Milvus的集合
             collection_name = milvus_config.chunks_collection

             # 4、处理 item_names 中的引号，防止注入或语法错误
             expr = None
             if item_names:
                 #quoted = ", ".join(f'"{v}"' for v in item_names)
                 #expr = f"item_name in [{quoted}]"
                 # 'item_name in ["BrotherHAK-180烫金机","BrotherHAK180烫金机"]'
                 expr = f'item_name in {item_names}'
                 logger.info(f"过滤条件: {expr}")
             else:
                 logger.info("未指定商品名过滤，将全库检索")

             # 5、构造Milvus混合搜索请求对象
             reqs = create_hybrid_search_requests(
                 dense_vector=dense_vec,
                 sparse_vector=sparse_vec,
                 dense_params=build_dense_params(strategy),
                 expr=expr,
                 limit=strategy.get("per_path_limit", 10)  # 底层检索返回数量（后续会再截断，预留更多结果做重排序）
             )

             # 6、执行混合向量检索
             logger.info("开始执行 Milvus 混合检索...")
             client = get_milvus_client()
             res = hybrid_search(
                 client=client,
                 collection_name=collection_name,  # 检索的目标集合名（文本片段向量集合）
                 reqs=reqs,  # 构造好的混合搜索请求对象（稠密+稀疏）
                 ranker_weights=(strategy.get("dense_weight", 0.8), strategy.get("sparse_weight", 0.2)),  # 稠/稀疏权重由意图策略驱动
                 limit=strategy.get("hybrid_limit", 5),  # 混合搜索最终返回数量
                 output_fields=["chunk_id", "content", "item_name", "title", "parent_title", "file_title", "part"]  # 业务字段（含评测三元组字段）
             )

             # 7、构造并返回结果：若检索结果非空，取res[0]，否则返回空列表
             # res 为 None = 这一路根本没查成（Milvus 连不上 / 查询报错），和「查了但没命中」
             # 必须在观测数据里分开：否则 Milvus 挂掉只会表现成召回为 0，值班会去查 embedding
             # 和索引，方向全错
             chunks = res[0] if res else []
             error = None if res is not None else "Milvus 无返回（服务不可用或查询失败）"

             # 8、本路观测数据：耗时/命中进 Prometheus，参数进 state 供慢查询日志排查
             elapsed_ms = (time.perf_counter() - started) * 1000
             if error:
                 # 失败时不记检索指标：把失败耗时和 0 命中记进直方图，会把 p95 拉高、把召回拉低，
                 # 两个指标同时失真。失败只进错误计数，细节留在 query_log 的 per_path 里
                 observe_node_error("query", self.name)
             else:
                 observe_retrieval("dense", elapsed_ms / 1000, len(chunks))
             params = {
                 "ef_search": strategy.get("ef_search"),
                 "per_path_limit": strategy.get("per_path_limit", 10),
                 "hybrid_limit": strategy.get("hybrid_limit", 5),
                 "dense_weight": strategy.get("dense_weight", 0.8),
                 "sparse_weight": strategy.get("sparse_weight", 0.2),
                 "item_filter": bool(item_names),
             }
             if error:
                 params["error"] = error
             return {
                 "embedding_chunks": chunks,
                 "dense_path_metric": path_metric("dense", hits=len(chunks), latency_ms=elapsed_ms,
                                                  params=params),
             }

         except Exception as e:
             logger.exception(f"向量搜索失败: {e}")
             elapsed_ms = (time.perf_counter() - started) * 1000
             # 这一路自己吞了异常，基类的错误计数不会触发，所以在这里补一次，
             # 否则「节点错误率」告警对本节点永远是 0
             observe_node_error("query", self.name)
             return {
                 "embedding_chunks": [],
                 "dense_path_metric": path_metric("dense", hits=0, latency_ms=elapsed_ms,
                                                  params={"error": str(e)[:200]}),
             }





if __name__ == "__main__":

    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？",
        "item_names": ["BrotherHAK180烫金机", "BrotherHAK-180烫金机"]
    }
    node_search_embedding = NodeSearchEmbedding()
    result = node_search_embedding(init_state)
    logger.info(serialize_json(result, indent=4))