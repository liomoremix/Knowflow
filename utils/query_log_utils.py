# utils/query_log_utils.py

"""查询日志（Mongo collection: query_log）。

用途：
1. 慢查询排查：记录每路检索的命中数与耗时，定位是哪一路拖慢了整体
2. P5 线上抽样评测：从真实查询里抽样，用 LLM 当裁判打相关性分
3. P3 索引一致性校验：用真实查询对比新旧两个物理集合的检索结果

设计约束：写日志绝不能拖垮或打断查询请求，因此所有异常只记录不抛出；
连接采用懒加载，Mongo 不可用时也不影响进程启动（区别于 mongo_history_utils 的 import 期建连）。
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pymongo import ASCENDING, DESCENDING

from tool.logger import logger
from utils.mongo_ops_utils import get_collection

load_dotenv()

COLLECTION_NAME = "query_log"

# 时间倒序查最新日志、按 slow 捞慢查询、按意图分组统计
INDEXES = [
    ([("ts", DESCENDING)], {}),
    ([("slow", ASCENDING), ("ts", DESCENDING)], {}),
    ([("intent", ASCENDING), ("ts", DESCENDING)], {}),
]


def get_query_log_collection():
    """
    获取 query_log 集合（懒加载 + 索引幂等创建）
    :return: pymongo Collection；Mongo 不可用时返回 None（调用方需容错）
    """
    return get_collection(COLLECTION_NAME, INDEXES)


def save_query_log(record: Dict[str, Any]) -> Optional[str]:
    """
    写入一条查询日志
    :param record: 日志字典，字段见 build_query_log
    :return: 记录 id；失败返回 None
    """
    collection = get_query_log_collection()
    if collection is None:
        return None
    try:
        result = collection.insert_one(record)
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"query_log 写入失败: {e}")
        return None


def build_query_log(
        session_id: str = "",
        message_id: str = "",
        original_query: str = "",
        rewritten_query: str = "",
        intent: str = "",
        intent_confidence: float = 0.0,
        intent_layer: str = "",
        strategy: Optional[dict] = None,
        item_names: Optional[list] = None,
        per_path: Optional[dict] = None,
        rrf_top: Optional[list] = None,
        reranked: Optional[list] = None,
        total_latency_ms: float = 0.0,
        slow: bool = False,
        collection_name: str = "",
) -> Dict[str, Any]:
    """
    组装查询日志文档。
    注意：不保存 1024 维查询向量（可随时重算、体积大、对排查无益），只保存检索参数。
    """
    return {
        "session_id": session_id,
        "message_id": message_id,
        "original_query": original_query,
        "rewritten_query": rewritten_query,
        "intent": intent,
        "intent_confidence": intent_confidence,
        "intent_layer": intent_layer,
        "strategy": strategy or {},
        "item_names": item_names or [],
        # 每路的命中数/耗时/参数，用于定位慢在哪一路：{"dense": {...}, "hyde": {...}, "web": {...}}
        "per_path": per_path or {},
        "rrf_top": rrf_top or [],
        # 最终给到大模型的切片：[{chunk_id, score, file_title, title, part}]
        "reranked": reranked or [],
        "total_latency_ms": total_latency_ms,
        "slow": slow,
        "collection_name": collection_name,
        "ts": datetime.now().timestamp(),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def query_logs(
        limit: int = 100,
        slow_only: bool = False,
        since: Optional[float] = None,
        intent: Optional[str] = None,
        projection: Optional[dict] = None,
) -> List[Dict[str, Any]]:
    """
    查询日志列表（按时间倒序）
    :param limit: 返回条数
    :param slow_only: 只看慢查询
    :param since: 只取该时间戳之后的记录
    :param intent: 只取该意图的记录
    :param projection: 字段投影，默认不返回大字段
    :return: 日志文档列表；失败返回空列表
    """
    collection = get_query_log_collection()
    if collection is None:
        return []

    conditions: Dict[str, Any] = {}
    if slow_only:
        conditions["slow"] = True
    if since:
        conditions["ts"] = {"$gte": since}
    if intent:
        conditions["intent"] = intent

    try:
        cursor = collection.find(conditions, projection).sort("ts", DESCENDING).limit(limit)
        return list(cursor)
    except Exception as e:
        logger.error(f"query_log 查询失败: {e}")
        return []


def sample_queries(
        limit: int = 30,
        since: Optional[float] = None,
        exclude_intents: Optional[list] = None,
        seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    随机抽样真实查询（P5 线上抽样评测用）
    :param limit: 抽样条数
    :param since: 只从该时间戳之后的记录里抽
    :param exclude_intents: 排除的意图（例如超范围闲聊没必要评检索质量）
    :param seed: 随机种子，$sample 不支持种子，改为本地洗牌
    :return: 日志文档列表
    """
    import random

    conditions: Dict[str, Any] = {}
    if since:
        conditions["ts"] = {"$gte": since}
    if exclude_intents:
        conditions["intent"] = {"$nin": exclude_intents}

    collection = get_query_log_collection()
    if collection is None:
        return []

    try:
        # 多取一些再本地洗牌：$sample 在大集合上会走随机游标，代价更高
        pool = list(collection.find(conditions).sort("ts", DESCENDING).limit(max(limit * 5, limit)))
        random.Random(seed).shuffle(pool)
        return pool[:limit]
    except Exception as e:
        logger.error(f"query_log 抽样失败: {e}")
        return []


def stats(since: Optional[float] = None) -> Dict[str, Any]:
    """
    汇总统计：总量、慢查询数与占比、各意图分布（供 P4 指标回填与 P5 判定）
    :param since: 只统计该时间戳之后的记录
    """
    collection = get_query_log_collection()
    if collection is None:
        return {}

    match = {"ts": {"$gte": since}} if since else {}
    try:
        rows = list(collection.aggregate([
            {"$match": match},
            {"$group": {
                "_id": "$intent",
                "count": {"$sum": 1},
                "slow_count": {"$sum": {"$cond": ["$slow", 1, 0]}},
                "avg_latency_ms": {"$avg": "$total_latency_ms"},
                "p99_latency_ms": {"$percentile": {
                    "input": "$total_latency_ms", "percentiles": [0.99], "method": "approximate"
                }},
            }},
        ]))
    except Exception as e:
        # $percentile 需要 MongoDB 7.0+，低版本退化为基础统计
        logger.warning(f"query_log 聚合统计失败，退化为计数: {e}")
        try:
            rows = list(collection.aggregate([
                {"$match": match},
                {"$group": {
                    "_id": "$intent",
                    "count": {"$sum": 1},
                    "slow_count": {"$sum": {"$cond": ["$slow", 1, 0]}},
                    "avg_latency_ms": {"$avg": "$total_latency_ms"},
                }},
            ]))
        except Exception as e2:
            logger.error(f"query_log 统计失败: {e2}")
            return {}

    by_intent = {}
    for row in rows:
        by_intent[row.get("_id") or "UNKNOWN"] = {
            "count": row.get("count", 0),
            "slow_count": row.get("slow_count", 0),
            "avg_latency_ms": round(row.get("avg_latency_ms") or 0.0, 2),
            "p99_latency_ms": round((row.get("p99_latency_ms") or [None])[0] or 0.0, 2),
        }

    total = sum(v["count"] for v in by_intent.values())
    slow_total = sum(v["slow_count"] for v in by_intent.values())
    return {
        "total": total,
        "slow_total": slow_total,
        "slow_rate": round(slow_total / total, 4) if total else 0.0,
        "by_intent": by_intent,
    }


if __name__ == "__main__":
    # 自测：写一条假日志再查出来（需要 Mongo 在跑）
    demo = build_query_log(
        session_id="selftest",
        original_query="HAK180烫金机怎么调温度",
        rewritten_query="BrotherHAK180烫金机 如何调节转印温度",
        intent="SEMANTIC",
        intent_confidence=0.91,
        intent_layer="L1",
        strategy={"dense_weight": 0.8, "sparse_weight": 0.2, "ef_search": 128},
        item_names=["BrotherHAK180烫金机"],
        per_path={"dense": {"hits": 10, "latency_ms": 42.1}, "hyde": {"hits": 8, "latency_ms": 910.4}},
        rrf_top=["c1", "c2", "c3"],
        reranked=[{"chunk_id": "c1", "score": 0.93, "file_title": "HAK180手册", "title": "温度调节", "part": 2}],
        total_latency_ms=1180.5,
        slow=True,
        collection_name="kb_chunks",
    )
    print("inserted:", save_query_log(demo))
    print("recent:", [r.get("original_query") for r in query_logs(limit=3)])
    print("slow:", len(query_logs(limit=10, slow_only=True)))
    print("stats:", stats())
