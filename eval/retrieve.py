# eval/retrieve.py

"""离线检索驱动器：直接复用线上节点，不改动业务代码。

评测必须跑「和线上完全一样的检索逻辑」，否则指标没有意义。这里不重新实现检索，
而是把 NodeSearchEmbedding / NodeSearchEmbeddingHyde / NodeRrf / NodeRerank 当成
普通函数调用（节点自带 __main__ 自测模式，本来就支持脱离 LangGraph 单独跑）。
"""

from contextlib import contextmanager
from typing import Optional

from config.milvus_config import milvus_config
from eval.config import eval_config
from processor.query_processor.nodes.node_rerank import NodeRerank
from processor.query_processor.nodes.node_rrf import NodeRrf
from processor.query_processor.nodes.node_search_embedding import NodeSearchEmbedding
from processor.query_processor.nodes.node_search_embedding_hyde import NodeSearchEmbeddingHyde
from tool.logger import logger

# 节点实例复用：节点内部无状态，避免每条查询重复构造
_dense_node = NodeSearchEmbedding()
_hyde_node = NodeSearchEmbeddingHyde()
_rrf_node = NodeRrf()
_rerank_node = NodeRerank()
_intent_node = None  # 懒加载：只有 use_intent=True 才需要（会加载 BGE-M3 / 调 LLM）


class EvalRetrievalError(RuntimeError):
    """检索节点内部异常（节点会吞掉异常返回空 dict，评测必须显式失败而不是记 0 分）"""


@contextmanager
def collection_override(collection: Optional[str]):
    """
    临时切换检索目标集合，用于 P3 蓝绿双索引的一致性对比
    :param collection: 目标物理集合名；None 表示沿用 .env 配置
    """
    if not collection:
        yield
        return
    original = milvus_config.chunks_collection
    milvus_config.chunks_collection = collection
    try:
        yield
    finally:
        milvus_config.chunks_collection = original


def resolve_strategy(intent: Optional[str] = None, ef_search: Optional[int] = None,
                     dense_weight: Optional[float] = None, sparse_weight: Optional[float] = None,
                     enable_hyde: Optional[bool] = None) -> dict:
    """
    组装评测用检索策略：以意图策略为底，允许单参数覆盖（调参/网格搜索用）
    :param intent: 意图类别，None 表示用默认策略（等同改造前行为）
    :param ef_search: 覆盖 HNSW ef
    :param dense_weight: 覆盖稠密权重（稀疏权重同步取补数，保证和为 1）
    :param sparse_weight: 覆盖稀疏权重
    :param enable_hyde: 覆盖 HyDE 开关
    :return: 策略字典
    """
    from config.intent_config import INTENT_DEFAULT, get_strategy_dict

    strategy = get_strategy_dict(intent or INTENT_DEFAULT)

    # 评测默认放宽召回数量：线上 topk=5 时 recall@10 恒等于 recall@5，指标失真
    strategy["per_path_limit"] = eval_config.eval_per_path_limit
    strategy["hybrid_limit"] = eval_config.eval_hybrid_limit

    if ef_search is not None:
        strategy["ef_search"] = ef_search
    if dense_weight is not None:
        strategy["dense_weight"] = dense_weight
        strategy["sparse_weight"] = round(1.0 - dense_weight, 6)
    if sparse_weight is not None:
        strategy["sparse_weight"] = sparse_weight
        strategy["dense_weight"] = round(1.0 - sparse_weight, 6)
    if enable_hyde is not None:
        strategy["enable_hyde"] = enable_hyde

    return strategy


def strategy_from_query(query: str, history: Optional[list] = None) -> tuple[dict, dict]:
    """
    走完整的三层意图漏斗得到策略（需要先加载 BGE-M3，可能调 LLM）
    :return: (策略字典, 意图节点输出的完整 state 增量)
    """
    global _intent_node
    if _intent_node is None:
        from processor.query_processor.nodes.node_intent_recognition import NodeIntentRecognition
        _intent_node = NodeIntentRecognition()

    state = {"original_query": query, "rewritten_query": query, "history": history or []}
    result = _intent_node.process(state)
    strategy = result.get("strategy") or resolve_strategy()
    # 意图节点已算好的查询向量直接带给检索节点，省一次 embedding
    strategy = dict(strategy)
    return strategy, result


def _run_dense(query: str, strategy: dict, item_names: Optional[list], reuse: Optional[dict]) -> list[dict]:
    """跑稠密+稀疏混合检索主路，返回 Milvus 命中列表"""
    state = {
        "rewritten_query": query,
        "item_names": item_names,
        "strategy": strategy,
    }
    if reuse:
        state["query_dense_vec"] = reuse.get("query_dense_vec")
        state["query_sparse_vec"] = reuse.get("query_sparse_vec")

    result = _dense_node.process(state)
    if "embedding_chunks" not in result:
        raise EvalRetrievalError(f"主检索路执行失败（节点内部异常），query={query!r}")
    return result["embedding_chunks"]


def _run_hyde(query: str, strategy: dict, item_names: Optional[list]) -> list[dict]:
    """跑 HyDE 路；策略关闭时节点内部直接早退，返回空列表"""
    if not strategy.get("enable_hyde", True):
        return []
    result = _hyde_node.process({
        "rewritten_query": query,
        "item_names": item_names,
        "strategy": strategy,
    })
    return result.get("hyde_embedding_chunks") or []


def _fuse(embedding_chunks: list[dict], hyde_chunks: list[dict]) -> list[dict]:
    """用线上同一个 RRF 节点融合两路结果"""
    state = _rrf_node.process({
        "embedding_chunks": embedding_chunks,
        "hyde_embedding_chunks": hyde_chunks,
    })
    return state.get("rrf_chunks") or []


def retrieve_raw(query: str, strategy: Optional[dict] = None, item_names: Optional[list] = None,
                 collection: Optional[str] = None, reuse: Optional[dict] = None) -> list[dict]:
    """
    只测召回层：混合检索（+HyDE）→ RRF，跳过 rerank
    :param query: 查询文本
    :param strategy: 检索策略，None 表示用默认策略
    :param item_names: 商品名过滤（模拟线上 item_name in [...]）
    :param collection: 指定物理集合，None 表示用 .env 配置的集合
    :param reuse: 意图节点产出的 state 增量，用于复用已算好的查询向量
    :return: 按 RRF 得分降序的切片字典列表（含 file_title/title/part/content 等字段）
    """
    strategy = strategy or resolve_strategy()
    with collection_override(collection):
        chunks = _run_dense(query, strategy, item_names, reuse)
        hyde_chunks = _run_hyde(query, strategy, item_names)
    return _fuse(chunks, hyde_chunks)


def retrieve(query: str, strategy: Optional[dict] = None, item_names: Optional[list] = None,
             collection: Optional[str] = None, use_rerank: bool = True,
             reuse: Optional[dict] = None) -> list[dict]:
    """
    测完整链路：召回层 → rerank 断崖截断，等同线上返回给大模型的那批切片
    :param use_rerank: False 时等价于 retrieve_raw（rerank 需要调用外部 DashScope 服务）
    :return: 最终切片字典列表
    """
    strategy = strategy or resolve_strategy()
    rrf_chunks = retrieve_raw(query, strategy, item_names, collection, reuse)
    if not use_rerank or not rrf_chunks:
        return rrf_chunks

    state = _rerank_node.process({
        "rewritten_query": query,
        "rrf_chunks": rrf_chunks,
        "web_search_docs": [],
        "strategy": strategy,
    })
    return state.get("reranked_docs") or []


def to_ranked_docs(chunks: list[dict]) -> list[dict]:
    """
    把节点输出统一成「含三元组字段的字典列表」，供 metrics 使用
    RRF 路的元素是 Milvus entity 字典；rerank 路的元素是合并后的结构化文档，
    两者都已带 file_title/title/part，这里只做字段裁剪，避免结果文件过大。
    """
    docs = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        docs.append({
            "file_title": chunk.get("file_title"),
            "title": chunk.get("title"),
            "part": chunk.get("part"),
            "item_name": chunk.get("item_name"),
            "chunk_id": chunk.get("chunk_id"),
            "content": chunk.get("content"),
            "score": chunk.get("score"),
            "source": chunk.get("source"),
        })
    return docs


def log_retrieval(query: str, docs: list[dict], limit: int = 3) -> None:
    """打印前若干条命中，便于人工排查评测异常"""
    logger.info(f"query={query!r} 命中前 {limit} 条:")
    for idx, doc in enumerate(docs[:limit], start=1):
        logger.info(f"  #{idx} ({doc.get('file_title')} | {doc.get('title')} | part={doc.get('part')})")
