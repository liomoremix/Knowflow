# processor/query_processor/state.py

from typing import TypedDict, List

class QueryGraphState(TypedDict):
    """
    查询流程图状态
    包含整个查询流程中传递的所有数据。
    """

    session_id: str  # 会话ID
    message_id: str  # 消息ID

    original_query: str  # 用户原始问题

    # 检索过程中的中间数据
    embedding_chunks: list  # 普通向量检索回来的切片
    hyde_embedding_chunks: list  # 已向量化的假设性问题切片
    hyde_doc: str  # HyDE 生成的假设性文档
    web_search_docs: list  # 网络搜索回来的文档

    # 排序过程中的数据
    rrf_chunks: list  # RRF 融合排序后的切片
    reranked_docs: list  # 重排序后的最终 Top-K 文档

    # 生成过程中的数据
    prompt: str  # 组装好的 Prompt
    answer: str  # 最终生成的答案

    # 辅助信息
    item_names: List[str]  # 提取出的商品名称
    rewritten_query: str  # 改写后的问题
    history: list  # 历史对话记录
    is_stream: bool  # 是否流式输出

    # 意图识别（三层漏斗）产出
    intent: str  # 意图类别：EXACT/SEMANTIC/AMBIGUOUS/OUT_OF_SCOPE/DEFAULT
    intent_confidence: float  # 意图置信度
    intent_layer: str  # 命中层级：L1/L2/L3/override/disabled
    strategy: dict  # 意图驱动的检索策略（权重/ef/开关/topk）
    query_dense_vec: list  # 查询稠密向量（意图节点算一次，检索节点复用）
    query_sparse_vec: dict  # 查询稀疏向量（同上）

    # API 可选覆盖参数
    intent_override: str  # 外部指定意图，非空跳过识别
    top_k_override: int  # 外部指定最终返回条数

    # 监控观测数据（P4）
    started_at: float  # 查询开始时间戳，由 KBQueryWorkflow.run 兜底写入，用于算总耗时
    # 三路检索各自的耗时/命中/参数。必须一路一个键：三路是并行分支，
    # 共用一个键会触发 LangGraph 的并发写冲突（InvalidUpdateError）
    dense_path_metric: dict  # 稠密+稀疏混合检索路
    hyde_path_metric: dict  # HyDE 假设文档检索路（耗时含 LLM 生成）
    web_path_metric: dict  # 联网搜索路（耗时含 MCP 往返）