# utils/query_trace_utils.py

"""一次查询的观测数据收集：单路耗时/命中 → 慢查询判定 → 落 query_log。

为什么单独一个模块：这些数据横跨「三路检索节点 + 重排节点 + 答案节点」，
放在任何一个节点里都会让它同时管业务和观测。这里只负责观测，节点只管往里塞。

两条硬约束：
· **绝不抛异常、绝不阻塞主链路**。观测失败最多少一条日志，不能让用户的查询挂掉。
· **不落大字段**。查询向量（1024 维）可随时重算、切片正文可回库查，写进 Mongo 只会让
  日志集合迅速膨胀到没法查。所以 per_path 只记参数与耗时，reranked 只记三元组与分数。
"""

import random
import time
from typing import Any, Dict, List, Optional

from config.milvus_config import milvus_config
from config.monitor_config import monitor_config
from tool.logger import logger
from utils.metrics_utils import observe_query, observe_slow_query
from utils.query_log_utils import build_query_log, save_query_log

# 三路检索各自往 state 里写自己的那一个键。
# 不能共用一个 "path_metrics" 键：三路是并行分支，LangGraph 对同一个键的并发写会直接报
# InvalidUpdateError（除非给状态字段挂 reducer），分开写最省事也最直观。
PATH_STATE_KEYS = {
    "dense": "dense_path_metric",
    "hyde": "hyde_path_metric",
    "web": "web_path_metric",
}


def path_metric(path: str,
                hits: Optional[int] = None,
                latency_ms: float = 0.0,
                params: Optional[dict] = None,
                skipped: bool = False) -> Dict[str, Any]:
    """
    组装一路检索的观测数据（既进 state 供落库，也用来判断单路是否慢）
    :param path: dense / hyde / web
    :param hits: 命中条数，None 表示这一路没结果或没跑
    :param latency_ms: 这一路的耗时（hyde 含 LLM 生成假设文档的时间，web 含 MCP 往返）
    :param params: 检索参数（ef/权重/limit），排查「为什么这一路慢」时要看它
    :param skipped: 被意图策略关掉了
    """
    return {
        "path": path,
        "hits": hits,
        "latency_ms": round(latency_ms, 1),
        "params": params or {},
        "skipped": skipped,
    }


def collect_per_path(state: Dict[str, Any]) -> Dict[str, Any]:
    """把三路各自写在 state 里的观测数据汇总成 {"dense": {...}, ...}"""
    per_path = {}
    for path, key in PATH_STATE_KEYS.items():
        metric = state.get(key)
        if metric:
            per_path[path] = metric
    return per_path


def total_latency_ms(state: Dict[str, Any]) -> float:
    """
    整条查询链路的耗时
    started_at 由 KBQueryWorkflow.run 兜底写入；缺了就返回 0（只按单路判慢）
    """
    started = state.get("started_at")
    if not started:
        return 0.0
    return round((time.time() - float(started)) * 1000, 1)


def slow_reasons(total_ms: float, per_path: Dict[str, Any]) -> List[str]:
    """
    判断慢在哪：整体慢 / 某一路慢
    两个条件是独立的：三路并行时「单路慢」几乎必然导致「整体慢」，但反过来不成立
    （LLM 生成慢时三路都很快，整体照样超阈值）
    :return: ["total"] / ["path"] / ["total", "path"] / []
    """
    reasons = []
    if total_ms and total_ms > monitor_config.slow_total_ms:
        reasons.append("total")
    for metric in (per_path or {}).values():
        if metric.get("skipped"):
            continue
        if (metric.get("latency_ms") or 0) > monitor_config.slow_path_ms:
            reasons.append("path")
            break
    return reasons


def should_log(slow: bool, sample_rate: Optional[float] = None) -> bool:
    """
    这条查询要不要写 query_log
    慢查询无条件写（排查靠它）；正常查询按抽样率写，给 P5 的线上评测提供无偏样本
    """
    if slow:
        return True
    rate = monitor_config.query_log_sample_rate if sample_rate is None else sample_rate
    if rate >= 1.0:
        return True
    if rate <= 0.0:
        return False
    return random.random() < rate


def _doc_briefs(docs: List[dict], limit: int = 10) -> List[dict]:
    """
    只留定位用的字段：业务三元组 + 分数 + 来源
    不留 content——正文能靠三元组回库查，留着只会把日志撑大
    """
    briefs = []
    for doc in (docs or [])[:limit]:
        briefs.append({
            "chunk_id": str(doc.get("chunk_id")) if doc.get("chunk_id") is not None else None,
            "score": doc.get("score"),
            "source": doc.get("source"),
            "file_title": doc.get("file_title"),
            "title": doc.get("title"),
            "part": doc.get("part"),
        })
    return briefs


def trace_query(state: Dict[str, Any], status: str = "ok") -> Optional[str]:
    """
    查询链路结束时调用一次：上报查询级指标 + 按策略落 query_log
    :param state: 最终状态
    :param status: ok / failed
    :return: query_log 记录 id；没写或写失败返回 None
    """
    try:
        intent = state.get("intent") or "UNKNOWN"
        total_ms = total_latency_ms(state)
        per_path = collect_per_path(state)

        observe_query(intent, status, total_ms / 1000.0 if total_ms else 0.0)

        reasons = slow_reasons(total_ms, per_path)
        slow = bool(reasons)
        for reason in reasons:
            observe_slow_query(reason)

        if not monitor_config.query_log_enabled or not should_log(slow):
            return None

        record = build_query_log(
            session_id=state.get("session_id") or "",
            message_id=state.get("message_id") or "",
            original_query=state.get("original_query") or "",
            rewritten_query=state.get("rewritten_query") or "",
            intent=intent,
            intent_confidence=state.get("intent_confidence") or 0.0,
            intent_layer=state.get("intent_layer") or "",
            strategy=state.get("strategy") or {},
            item_names=state.get("item_names") or [],
            per_path=per_path,
            rrf_top=_doc_briefs(state.get("rrf_chunks") or []),
            reranked=_doc_briefs(state.get("reranked_docs") or []),
            total_latency_ms=total_ms,
            slow=slow,
            collection_name=milvus_config.chunks_collection or "",
        )
        # 慢在哪一路直接记进去，排查时不用再对着 per_path 逐条比阈值
        if reasons:
            record["slow_reasons"] = reasons

        log_id = save_query_log(record)
        if slow:
            logger.warning(f"慢查询：总耗时 {total_ms}ms，原因 {reasons}，"
                           f"各路 {[{k: v.get('latency_ms') for k, v in per_path.items()}]}")
        return log_id
    except Exception as e:
        # 观测失败不能影响已经把答案返回给用户的请求
        logger.error(f"查询观测数据落库失败: {e}")
        return None
