# config/index_config.py

"""索引选型、参数与蓝绿双索引配置。

现状：建集合时稠密向量用 AUTOINDEX（Milvus 自行选择索引类型与参数），既不可控也没法调优。
改造后：显式 HNSW + 明确的 M / ef_construction，并支持在不动线上的前提下重建索引、
校验一致、原子切换（alias）、失败回滚。

注意：运行期各意图用多大的 ef 去搜，定义在 config/intent_config.py 的策略表里，
这里只管「索引怎么建」和「重建/校验/切换的规则」，避免同一份参数出现在两处。
"""

import os
from dataclasses import dataclass, field, asdict

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    """读整型环境变量，缺失或非法时回退默认值"""
    raw = os.getenv(name)
    try:
        return int(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """读浮点环境变量，缺失或非法时回退默认值"""
    raw = os.getenv(name)
    try:
        return float(raw) if raw not in (None, "") else default
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    """读布尔环境变量，缺失或非法时回退默认值"""
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class DenseIndexSpec:
    """稠密向量索引规格。

    为什么选 HNSW：知识库是「文档量中等（万级~百万级）、要求低延迟高召回、可以接受
    较高内存占用」的典型场景，HNSW 在这类场景下召回/延迟综合表现最好，也是 Milvus
    官方对内存型部署的默认推荐。IVF 系列更省内存但召回需要靠 nprobe 换，调参更麻烦。
    """
    index_type: str = "HNSW"
    metric_type: str = "COSINE"  # 与 BGE-M3 稠密向量的归一化特性匹配
    # M：每个节点的最大出边数。越大召回越高、内存与建索引时间越多。32 是通用起点。
    M: int = 32
    # ef_construction：建图时的候选队列长度。越大图质量越高、建索引越慢。256 是通用起点。
    ef_construction: int = 256

    def index_params(self) -> dict:
        """转成 Milvus add_index 需要的 params 字典"""
        return {"M": self.M, "efConstruction": self.ef_construction}


@dataclass
class SparseIndexSpec:
    """稀疏向量索引规格（BGE-M3 的词权重向量）"""
    index_type: str = "SPARSE_INVERTED_INDEX"
    metric_type: str = "IP"
    inverted_index_algo: str = "DAAT_MAXSCORE"  # 倒排遍历算法，MaxScore 剪枝更快
    normalize: bool = True
    quantization: str = "none"

    def index_params(self) -> dict:
        return {
            "inverted_index_algo": self.inverted_index_algo,
            "normalize": self.normalize,
            "quantization": self.quantization,
        }


@dataclass
class IndexConfig:
    """索引管理与蓝绿切换配置"""

    # ---- 命名 ----
    # 逻辑集合名 = alias 名 = .env 的 CHUNKS_COLLECTION，查询侧永远只认这个名字
    logical_collection: str = os.getenv("CHUNKS_COLLECTION") or "kb_chunks"
    # 物理集合命名规则：{logical}_v{version}，例如 kb_chunks_v2
    physical_prefix: str = ""  # 空表示用 logical_collection 作前缀

    # ---- 索引参数 ----
    dense_index: DenseIndexSpec = field(default_factory=lambda: DenseIndexSpec(
        M=_env_int("HNSW_M", 32),
        ef_construction=_env_int("HNSW_EF_CONSTRUCTION", 256),
        metric_type=os.getenv("MILVUS_METRIC_TYPE") or "COSINE",
    ))
    sparse_index: SparseIndexSpec = field(default_factory=SparseIndexSpec)

    # ---- 调参网格（benchmark.py 用）----
    # 跑网格搜索时逐个组合建临时集合、用评测集测 recall/latency，最后定参
    bench_M: list = field(default_factory=lambda: [16, 32, 48])
    bench_ef_construction: list = field(default_factory=lambda: [128, 256])
    bench_ef_search: list = field(default_factory=lambda: [32, 64, 128, 256])
    bench_sample_limit: int = 200  # 网格搜索时每场景最多跑多少条评测样本（控成本）

    # ---- 蓝绿双索引 ----
    # 重建期间是否把新导入的数据同时写进新旧两个集合（双写），保证切换时新集合不缺数据
    double_write_enabled: bool = _env_bool("INDEX_DOUBLE_WRITE_ENABLED", True)
    backfill_batch_size: int = _env_int("INDEX_BACKFILL_BATCH", 200)
    # move_vectors：直接把旧集合里的向量搬到新集合（快，不花模型算力，要求 embedding 模型没变）
    # re_embed：读旧集合的 content 重新算向量（慢且花钱，embedding 模型换了才用）
    default_rebuild_mode: str = os.getenv("INDEX_REBUILD_MODE") or "move_vectors"
    # 旧索引切换后保留多少天再删（留出回滚窗口）
    retire_after_days: int = _env_int("INDEX_RETIRE_AFTER_DAYS", 7)

    # ---- 一致性校验阈值（validate.py 用）----
    validate_sample_size: int = _env_int("INDEX_VALIDATE_SAMPLES", 50)
    # 新旧集合 gold 命中率差不能超过这个值（1 个百分点）
    validate_hit_rate_max_diff: float = _env_float("INDEX_VALIDATE_HIT_DIFF", 0.01)
    # top10 结果三元组集合的 Jaccard 相似度下限
    validate_jaccard_min: float = _env_float("INDEX_VALIDATE_JACCARD", 0.85)
    # move_vectors 模式是逐条搬运，结果应当几乎完全一致，门槛更严
    validate_jaccard_min_move: float = _env_float("INDEX_VALIDATE_JACCARD_MOVE", 0.95)

    # ---- 索引质量代理指标阈值（P3 告警 / P5 自动重建判定用）----
    # sealed segment 太多说明碎片化严重，查询要扫的段变多、延迟上升，需要重建合并
    max_sealed_segments: int = _env_int("INDEX_MAX_SEALED_SEGMENTS", 64)
    # p99 延迟相比基线劣化超过这个倍数就算异常
    p99_degrade_ratio: float = _env_float("INDEX_P99_DEGRADE_RATIO", 1.5)

    # ---- 巡检调度器（P5：自动发现问题 + 一键触发重建）----
    scheduler_interval_hours: int = _env_int("SCHEDULER_INTERVAL_HOURS", 6)
    # 判定出「该重建」后要不要真的动手。默认 False：只出结论和命令，人来决定
    scheduler_auto_rebuild: bool = _env_bool("SCHEDULER_AUTO_REBUILD", False)
    # 自动触发时只做到 prepare_only（建新集合 + 双写 + 一致性校验 + 召回门禁，停在 ready）。
    # 切别名是流量切换，始终留给人确认——线上事故里最贵的就是「自动切换切错了」
    scheduler_prepare_only: bool = _env_bool("SCHEDULER_PREPARE_ONLY", True)
    # 自动触发时门禁评测每场景取样条数（比人工跑小，控制耗时）
    scheduler_eval_limit: int = _env_int("SCHEDULER_EVAL_LIMIT", 20)
    # 每轮线上抽样条数（0 表示跳过线上评测，只看索引侧信号）
    scheduler_online_sample: int = _env_int("SCHEDULER_ONLINE_SAMPLE", 30)

    def physical_name(self, version: int) -> str:
        """版本号 → 物理集合名"""
        prefix = self.physical_prefix or self.logical_collection
        return f"{prefix}_v{version}"

    def dense_index_dict(self) -> dict:
        return asdict(self.dense_index)

    def sparse_index_dict(self) -> dict:
        return asdict(self.sparse_index)


index_config = IndexConfig()
