# config/monitor_config.py

"""监控与慢查询日志配置。

指标全部走 Prometheus 拉取模型：进程内只负责「记」，聚合与告警在 Prometheus/Alertmanager 侧做，
所以这里没有采样窗口之类的东西，只有「什么算慢」「要不要记」这类业务阈值。

两个总开关的意义：
· METRICS_ENABLED=false —— 指标埋点全部退化成空操作（不建 Prometheus 对象、不占内存），
  用于怀疑埋点本身影响性能时的对照排查。
· QUERY_LOG_ENABLED=false —— 不往 Mongo 写 query_log，P5 的线上抽样评测会随之失去数据源，
  但查询主链路完全不受影响。
"""

import os
from dataclasses import dataclass, field
from typing import Tuple

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    """读布尔环境变量，缺失或非法时回退默认值"""
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    """读浮点环境变量，缺失或非法时回退默认值"""
    raw = os.getenv(name)
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


@dataclass
class MonitorConfig:
    """性能监控与慢查询日志阈值"""

    # ---- 开关 ----
    metrics_enabled: bool = field(default_factory=lambda: _env_bool("METRICS_ENABLED", True))
    query_log_enabled: bool = field(default_factory=lambda: _env_bool("QUERY_LOG_ENABLED", True))

    # ---- 慢查询判定 ----
    # 整条查询链路超过这个毫秒数就算慢查询（含 LLM 生成，所以给到秒级）
    slow_total_ms: float = field(default_factory=lambda: _env_float("SLOW_QUERY_TOTAL_MS", 3000.0))
    # 单路检索（dense/hyde/web）超过这个毫秒数就算慢：三路并行时总耗时由最慢的一路决定，
    # 只记总耗时会看不出是哪一路拖的，所以要按路记
    slow_path_ms: float = field(default_factory=lambda: _env_float("SLOW_QUERY_PATH_MS", 1000.0))

    # ---- query_log 写入策略 ----
    # 慢查询永远写；非慢查询按这个比例抽样写（1.0=全写）。
    # 抽样是为了 P5 的线上评测：只留慢查询会让样本偏向坏情况，评出来的相关率没有代表性。
    query_log_sample_rate: float = field(default_factory=lambda: _env_float("QUERY_LOG_SAMPLE_RATE", 1.0))

    # ---- 直方图分桶（秒）----
    # 覆盖「向量检索几十毫秒」到「LLM 生成十几秒」的全跨度；分桶太密会让 /metrics 体积暴涨
    latency_buckets: Tuple[float, ...] = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0)
    # 命中条数分桶：单路上限就是 per_path_limit，不需要大桶
    hits_buckets: Tuple[float, ...] = (0, 1, 2, 3, 5, 8, 10, 15, 20, 30, 50)

    # ---- p99 劣化判定基线（P5 自动重建用）----
    # 0 表示不判定；有值时线上 p99 超过 base * index_config.p99_degrade_ratio 即视为索引劣化
    p99_baseline_ms: float = field(default_factory=lambda: _env_float("LATENCY_P99_BASELINE_MS", 0.0))


monitor_config = MonitorConfig()
