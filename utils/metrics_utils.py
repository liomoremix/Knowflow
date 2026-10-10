# utils/metrics_utils.py

"""Prometheus 指标埋点。

三条设计约束（改这个文件前先读）：

1. **埋点绝不能影响业务链路**。所有对外函数都吞异常；prometheus_client 没装或指标被
   METRICS_ENABLED=false 关掉时，函数退化成空操作。监控自己挂掉不该让查询失败。

2. **标签基数必须可控**。只用有限枚举做标签（flow/node/path/intent/status/collection）。
   绝不把 task_id、session_id、query 文本这类无界值放进标签——每条新值都会在 Prometheus
   里生成一条新时间序列，内存和磁盘会被撑爆，这是监控最常见的翻车方式。
   想看单个任务的进度用日志或 /status 接口，不要用指标。

3. **跨进程状态用 Collector 现算，不用进程内 Gauge**。索引注册表在 Mongo、评测结果在文件里，
   进程内 Gauge 是「谁最后 set 谁说了算」，重启就归零、多进程互相覆盖。改成自定义 Collector
   在被抓取的那一刻去读权威数据源，值永远是真的。
"""

import json
import time
from typing import Optional

from config.monitor_config import monitor_config
from tool.logger import logger

try:
    from prometheus_client import REGISTRY, Counter, Histogram
    from prometheus_client.core import GaugeMetricFamily
    _PROMETHEUS_AVAILABLE = True
except ImportError:  # 没装 prometheus_client 时全部埋点变空操作
    _PROMETHEUS_AVAILABLE = False

_ENABLED = monitor_config.metrics_enabled and _PROMETHEUS_AVAILABLE

if not _PROMETHEUS_AVAILABLE:
    logger.warning("prometheus_client 未安装，指标埋点已禁用")
elif not monitor_config.metrics_enabled:
    logger.info("METRICS_ENABLED=false，指标埋点已禁用")

_BUCKETS = monitor_config.latency_buckets
_HITS_BUCKETS = monitor_config.hits_buckets


def _make_histogram(name: str, doc: str, labels: tuple, buckets=_BUCKETS):
    """建直方图；禁用时返回 None（调用方统一判空）"""
    if not _ENABLED:
        return None
    return Histogram(name, doc, labels, buckets=buckets)


def _make_counter(name: str, doc: str, labels: tuple):
    """建计数器；禁用时返回 None"""
    if not _ENABLED:
        return None
    return Counter(name, doc, labels)


# ==================== 查询链路指标 ====================
# flow=query/import：两条链路的节点分开看，导入慢和查询慢是两种完全不同的故障
NODE_LATENCY = _make_histogram(
    "kb_node_latency_seconds", "单个图节点的执行耗时", ("flow", "node"))
NODE_ERRORS = _make_counter(
    "kb_node_errors_total", "单个图节点的执行失败次数", ("flow", "node"))

# path=dense/hyde/web：三路并行，只记总耗时看不出是哪一路拖的（图中「分片监控」的落地形式）
RETRIEVAL_LATENCY = _make_histogram(
    "kb_retrieval_latency_seconds", "单路检索耗时", ("path",))
RETRIEVAL_HITS = _make_histogram(
    "kb_retrieval_hits", "单路检索命中条数", ("path",), buckets=_HITS_BUCKETS)

QUERY_LATENCY = _make_histogram(
    "kb_query_latency_seconds", "整条查询链路耗时（含 LLM 生成）", ("intent",))
QUERY_TOTAL = _make_counter(
    "kb_query_total", "查询次数（status=ok/failed），配 intent 可算 QPS 与错误率", ("intent", "status"))
SLOW_QUERY_TOTAL = _make_counter(
    "kb_slow_query_total", "慢查询次数（reason=total 整体慢 / path 单路慢）", ("reason",))
RERANK_OUTPUT_SIZE = _make_histogram(
    "kb_rerank_output_size", "重排断崖截断后交给 LLM 的切片条数", (),
    buckets=(0, 1, 2, 3, 5, 8, 10, 15, 20))

# ==================== 导入链路指标 ====================
IMPORT_TASKS = _make_counter(
    "kb_import_tasks_total", "导入任务结束次数（status=completed/failed）", ("status",))


def observe_node(flow: str, node: str, seconds: float) -> None:
    """记录一个节点的耗时（在两个 base 的 __call__ 里调用）"""
    if NODE_LATENCY is None:
        return
    try:
        NODE_LATENCY.labels(flow=flow, node=node).observe(max(seconds, 0.0))
    except Exception as e:
        logger.debug(f"记录节点耗时失败: {e}")


def observe_node_error(flow: str, node: str) -> None:
    """记录一个节点的失败次数"""
    if NODE_ERRORS is None:
        return
    try:
        NODE_ERRORS.labels(flow=flow, node=node).inc()
    except Exception as e:
        logger.debug(f"记录节点错误失败: {e}")


def observe_retrieval(path: str, seconds: float, hits: Optional[int] = None) -> None:
    """
    记录一路检索的耗时与命中数
    :param path: dense / hyde / web
    :param hits: 命中条数，None 表示这一路没跑（被意图策略关掉）
    """
    if RETRIEVAL_LATENCY is None:
        return
    try:
        RETRIEVAL_LATENCY.labels(path=path).observe(max(seconds, 0.0))
        if hits is not None and RETRIEVAL_HITS is not None:
            RETRIEVAL_HITS.labels(path=path).observe(hits)
    except Exception as e:
        logger.debug(f"记录检索指标失败: {e}")


def observe_query(intent: str, status: str, seconds: float) -> None:
    """记录一次完整查询：计数（算 QPS/错误率）+ 耗时（算 p99）"""
    if QUERY_TOTAL is None:
        return
    try:
        QUERY_TOTAL.labels(intent=intent or "UNKNOWN", status=status).inc()
        if QUERY_LATENCY is not None:
            QUERY_LATENCY.labels(intent=intent or "UNKNOWN").observe(max(seconds, 0.0))
    except Exception as e:
        logger.debug(f"记录查询指标失败: {e}")


def observe_slow_query(reason: str) -> None:
    """记录一次慢查询（reason=total/path）"""
    if SLOW_QUERY_TOTAL is None:
        return
    try:
        SLOW_QUERY_TOTAL.labels(reason=reason).inc()
    except Exception as e:
        logger.debug(f"记录慢查询失败: {e}")


def observe_rerank_size(count: int) -> None:
    """记录重排后实际交给 LLM 的切片条数（长期为 0 说明召回或重排出了问题）"""
    if RERANK_OUTPUT_SIZE is None:
        return
    try:
        RERANK_OUTPUT_SIZE.observe(max(count, 0))
    except Exception as e:
        logger.debug(f"记录重排条数失败: {e}")


def observe_import_task(status: str) -> None:
    """记录一个导入任务的最终状态"""
    if IMPORT_TASKS is None:
        return
    try:
        IMPORT_TASKS.labels(status=status).inc()
    except Exception as e:
        logger.debug(f"记录导入指标失败: {e}")


class _Timer:
    """极简计时器：with 块结束时自动上报耗时（成功失败都报，失败另外计数）"""

    def __init__(self, flow: str, node: str):
        self.flow = flow
        self.node = node
        self._started = 0.0

    def __enter__(self):
        self._started = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        observe_node(self.flow, self.node, time.perf_counter() - self._started)
        if exc_type is not None:
            observe_node_error(self.flow, self.node)
        return False  # 不吞异常，交给调用方处理


def node_timer(flow: str, node: str) -> "_Timer":
    """给 base.__call__ 用的计时上下文"""
    return _Timer(flow, node)


# ==================== 抓取时现算的运维指标 ====================

# 采集结果缓存有效期。/metrics 每 15s 被抓一次，但索引行数、重建状态这些量的变化是
# 以分钟计的，每次抓取都去问 Milvus + Mongo 属于白问；更要紧的是这两个依赖挂掉时，
# 抓取会被连接超时拖长到超过 Prometheus 的 scrape_timeout，target 被判 down，
# 「监控系统自己」反而成了告警源。所以现算 + 短 TTL 缓存。
_CACHE_TTL_SECONDS = 30.0

# 找「每种 mode 最新一份评测结果」时最多往回翻多少个文件：结果文件会越攒越多，
# 不设上限的话每次抓取都要 stat + 解析整个目录
MAX_RESULT_FILES = 20


class OpsCollector:
    """索引/重建/评测/导入进度指标：在 Prometheus 抓取的那一刻去读权威数据源。

    为什么不用普通 Gauge：这些数据分别活在 Mongo（注册表）、Milvus（行数/segment）、
    文件（评测结果）、进程内存（导入任务）里，用 set() 维护只会得到一个「上次谁写的」的假值。
    """

    def __init__(self):
        self._cache = []
        self._cached_at = 0.0

    def collect(self):
        """Prometheus 抓取时回调。任何异常都必须吞掉，否则整个 /metrics 端点会 500"""
        now = time.monotonic()
        if self._cache and now - self._cached_at < _CACHE_TTL_SECONDS:
            yield from self._cache
            return

        # 注册表状态一次抓取只问一遍：索引指标和重建指标都要用它
        status = self._safe(self._registry_status, {})
        families = []
        families.extend(self._safe(self._index_metrics, [], status))
        families.extend(self._safe(self._rebuild_metrics, [], status))
        families.extend(self._safe(self._eval_metrics, []))
        families.extend(self._safe(self._import_metrics, []))

        # 全挂了才沿用上一轮：宁可给旧值，也不能让 /metrics 变空（那样连 up 之外的
        # 所有业务指标一起消失，排查时等于蒙眼）。有部分结果就正常缓存这部分
        if families:
            self._cache = families
            self._cached_at = now
        else:
            families = self._cache
        yield from families

    @staticmethod
    def _safe(fn, default, *args):
        """
        每个数据源各自兜异常：prometheus_client 不替采集器兜，collect 里冒出一个异常
        整个端点就 500。Mongo 挂了不该让评测召回指标一起消失，反之亦然
        """
        try:
            return fn(*args)
        except Exception as e:
            logger.debug(f"采集 {getattr(fn, '__name__', '运维指标')} 失败: {e}")
            return default

    @staticmethod
    def _registry_status() -> dict:
        """读索引注册表（Mongo 不可用时返回空，指标就少几条序列，不影响其它指标）"""
        try:
            from utils import index_registry_utils as registry

            return registry.registry_status() or {}
        except Exception as e:
            logger.debug(f"读取索引注册表失败: {e}")
            return {}

    # ---- 索引规模与碎片化（图中「索引大小」监控）----
    def _index_metrics(self, status: dict):
        rows = GaugeMetricFamily("kb_index_rows", "集合行数（按注册表角色区分）",
                                 labels=["collection", "role"])
        segments = GaugeMetricFamily("kb_index_sealed_segments", "sealed segment 数（碎片化程度）",
                                     labels=["collection"])
        try:
            from processor.index_manager.quality import segment_stats
            from utils.milvus_utils import get_milvus_client, get_row_count

            client = get_milvus_client()
            for role in ("active", "building"):
                record = status.get(role) or {}
                physical = record.get("physical")
                if not physical or client is None:
                    continue
                count = get_row_count(client, physical)
                if count >= 0:
                    rows.add_metric([physical, role], count)
                # 段数只在 active 上采：在建集合还没载入，问了也是白问
                if role == "active":
                    sealed = (segment_stats(physical) or {}).get("sealed")
                    if sealed is not None:
                        segments.add_metric([physical], sealed)
        except Exception as e:
            logger.debug(f"采集索引规模指标失败: {e}")
        return [rows, segments]

    # ---- 蓝绿重建状态（图中「重建失败」告警的数据源）----
    def _rebuild_metrics(self, status: dict):
        family = GaugeMetricFamily("kb_rebuild_status", "各版本的注册表状态（1=当前处于该状态）",
                                   labels=["logical", "version", "status"])
        try:
            logical = status.get("logical") or ""
            for record in status.get("history") or []:
                version = record.get("version")
                state = record.get("status")
                if version is None or not state:
                    continue
                # 只有当前状态置 1，其余状态不产出序列：否则每个版本会拖着一堆恒 0 的序列
                family.add_metric([logical, str(version), state], 1)
        except Exception as e:
            logger.debug(f"采集重建状态指标失败: {e}")
        return [family]

    # ---- 评测召回率（图中「抽样召回率」监控；含 P5 的线上相关率）----
    def _eval_metrics(self):
        family = GaugeMetricFamily("kb_eval_recall", "最近一次评测的 recall@k",
                                   labels=["scenario", "k", "mode"])
        try:
            from eval.config import eval_config

            result_dir = eval_config.result_dir
            files = sorted(result_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
            # 每种 mode 各取最新一份，而不是「全局最新一份」：离线评测（full_chain/recall_layer）
            # 和线上抽样（online）是两套口径，只取最新文件会让线上评测一跑就把离线指标顶掉，
            # 看板与 RecallDrop 告警随之断流。历史值由 Prometheus 自己的时序存着
            seen_modes = set()
            for path in files[:MAX_RESULT_FILES]:
                if len(seen_modes) >= 3:  # 目前只有 full_chain / recall_layer / online 三种
                    break
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        result = json.load(f)
                except Exception as e:
                    # 单份结果文件坏了不该让其它评测指标一起消失
                    logger.debug(f"读取评测结果 {path.name} 失败: {e}")
                    continue
                mode = (result.get("meta") or {}).get("mode") or "unknown"
                if mode in seen_modes:
                    continue
                seen_modes.add(mode)
                scopes = dict(result.get("by_scenario") or {})
                if result.get("overall"):
                    scopes["overall"] = result["overall"]
                for scenario, metrics in scopes.items():
                    for key, value in (metrics or {}).items():
                        if not key.startswith("recall@") or value is None:
                            continue
                        family.add_metric([scenario, key.split("@", 1)[1], mode], float(value))
        except Exception as e:
            logger.debug(f"采集评测指标失败: {e}")
        return [family]

    # ---- 导入并发（不用 task_id 做标签，见文件头的基数说明）----
    def _import_metrics(self):
        family = GaugeMetricFamily("kb_import_running_tasks", "正在处理中的导入任务数")
        try:
            from utils import task_utils

            # 任务表是进程内存：只有跑导入的那个进程能看见自己的任务，这正是我们想要的
            running = getattr(task_utils, "_tasks", {}) or {}
            count = sum(1 for t in running.values()
                        if t.get("status") == task_utils.TASK_STATUS_PROCESSING)
            family.add_metric([], count)
        except Exception as e:
            logger.debug(f"采集导入进度失败: {e}")
        return [family]


_registered = False


def register_ops_collector() -> None:
    """注册运维指标采集器（幂等，在服务启动时调用一次）"""
    global _registered
    if not _ENABLED or _registered:
        return
    try:
        REGISTRY.register(OpsCollector())
        _registered = True
        logger.info("运维指标采集器已注册（索引规模/重建状态/评测召回/导入并发）")
    except Exception as e:
        # 重复注册等异常都不该影响服务启动
        logger.warning(f"运维指标采集器注册失败（其余埋点仍可用）: {e}")


def metrics_enabled() -> bool:
    """埋点是否生效（供健康检查/自检使用）"""
    return _ENABLED


def snapshot() -> dict:
    """把当前进程的关键指标读成字典（自测与 /health 用，不依赖 Prometheus 抓取）"""
    if not _ENABLED:
        return {"enabled": False}

    from prometheus_client import REGISTRY as _registry

    def value(name: str, **labels) -> float:
        """用公开 API 读单个样本值，读不到就是 0"""
        try:
            return float(_registry.get_sample_value(name, labels) or 0.0)
        except Exception:
            return 0.0

    out = {"enabled": True, "queries": {}, "retrieval": {}}
    for path in ("dense", "hyde", "web"):
        count = value("kb_retrieval_latency_seconds_count", path=path)
        total = value("kb_retrieval_latency_seconds_sum", path=path)
        out["retrieval"][path] = {
            "count": count,
            "avg_seconds": round(total / count, 4) if count else 0.0,
            "hits_avg": round(value("kb_retrieval_hits_sum", path=path) / count, 2) if count else 0.0,
        }
    for intent in ("EXACT", "SEMANTIC", "AMBIGUOUS", "OUT_OF_SCOPE", "DEFAULT", "UNKNOWN"):
        ok = value("kb_query_total", intent=intent, status="ok")
        failed = value("kb_query_total", intent=intent, status="failed")
        if ok or failed:
            out["queries"][intent] = {"ok": ok, "failed": failed}
    return out


if __name__ == "__main__":
    # 自测：打一批假数据，再从 REGISTRY 里读回来核对
    register_ops_collector()
    observe_node("query", "node_rerank", 0.42)
    observe_node_error("query", "node_rerank")
    observe_retrieval("dense", 0.05, 10)
    observe_query("EXACT", "ok", 1.2)
    observe_slow_query("path")
    observe_rerank_size(4)
    observe_import_task("completed")
    print("enabled:", metrics_enabled())
    print("snapshot:", snapshot())

    if _ENABLED:
        from prometheus_client import generate_latest

        text = generate_latest(REGISTRY).decode("utf-8")
        wanted = ["kb_node_latency_seconds", "kb_node_errors_total", "kb_retrieval_hits",
                  "kb_query_total", "kb_slow_query_total", "kb_rerank_output_size",
                  "kb_import_tasks_total", "kb_index_rows", "kb_rebuild_status",
                  "kb_eval_recall", "kb_import_running_tasks"]
        missing = [name for name in wanted if name not in text]
        print("缺失指标:", missing or "无")
        print("输出行数:", len(text.splitlines()))
