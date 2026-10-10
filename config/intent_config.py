# config/intent_config.py

import os
from dataclasses import asdict, dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool) -> bool:
    """读布尔环境变量，缺失或非法时回退默认值"""
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")

# 意图类别枚举
INTENT_EXACT = "EXACT"  # 精确查询：型号/参数/规格/价格等词面匹配为主的查询
INTENT_SEMANTIC = "SEMANTIC"  # 语义查询：场景/做法/原理/排障等需要语义理解的查询
INTENT_AMBIGUOUS = "AMBIGUOUS"  # 歧义查询：对比/多选/一词多义
INTENT_OUT_OF_SCOPE = "OUT_OF_SCOPE"  # 超范围/闲聊：知识库无法回答，短路直答
INTENT_DEFAULT = "DEFAULT"  # 意图识别关闭或兜底失败时的默认策略（等同改造前行为）


@dataclass
class IntentStrategy:
    """意图驱动的检索策略：融合权重、索引搜索参数、各路开关与截断参数"""
    dense_weight: float  # 稠密向量权重（WeightedRanker 第一位）
    sparse_weight: float  # 稀疏向量权重（WeightedRanker 第二位）
    ef_search: int  # HNSW 搜索参数 ef，0 表示不显式传参（沿用索引默认）
    per_path_limit: int  # 单向量路（dense/sparse）底层返回数量
    hybrid_limit: int  # 混合搜索融合后返回数量
    enable_hyde: bool  # 是否执行 HyDE 假设文档检索路
    enable_web_search: bool  # 是否执行联网搜索路
    rerank_max_topk: int  # 重排断崖截断硬上限
    rerank_min_topk: int  # 重排断崖截断保底条数

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class IntentConfig:
    intent_enabled: bool  # 总开关：False 时意图节点直通默认策略，行为等同改造前
    l1_keywords: dict  # 类别 -> 关键词列表（子串命中）
    l1_patterns: dict  # 类别 -> 正则列表（型号模式等）
    l2_prototypes: dict  # 类别 -> 意图原型句列表（BGE-M3 向量预缓存后做余弦相似度）
    l2_high: float  # L2 最高分达到该值且与次高分差足够时才直接定类
    l2_margin: float  # L2  top1 与 top2 的最小分差
    strategies: dict  # 类别 -> IntentStrategy
    out_of_scope_answer: str  # 超范围短路直答模板


intent_config = IntentConfig(
    # 关掉它就是改造前的行为，出问题时的第一回退手段
    intent_enabled=_env_bool("INTENT_ENABLED", True),
    l1_keywords={
        INTENT_OUT_OF_SCOPE: [
            "你好", "您好", "在吗", "你是谁", "你叫什么", "你能做什么", "你会什么",
            "谢谢", "多谢", "再见", "聊聊天", "讲个笑话", "天气怎么样",
        ],
        INTENT_AMBIGUOUS: ["对比", "比较", "区别", "差异", "哪个好", "哪个更", " versus ", " vs "],
        INTENT_EXACT: ["型号", "参数", "规格", "价格", "多少钱", "库存", "尺寸", "重量", "功率", "电压", "配件编号"],
        INTENT_SEMANTIC: ["怎么", "如何", "为什么", "原理", "方法", "步骤", "流程", "怎么办", "解决", "故障", "保养", "清洁"],
    },
    l1_patterns={
        INTENT_EXACT: [r"[A-Za-z]{2,}[- ]?\d{3,}"],  # 型号模式：HAK180 / P60 / TK-880 等
    },
    l2_prototypes={
        INTENT_EXACT: [
            "HAK180烫金机的技术参数是多少",
            "这个设备的规格和型号是什么",
            "产品的额定电压和功率是多少",
            "这个型号多少钱，有没有库存",
            "设备的尺寸和重量参数",
        ],
        INTENT_SEMANTIC: [
            "如何调节转印温度",
            "设备出现故障了应该怎么处理",
            "操作步骤和注意事项是什么",
            "为什么打印出来的颜色偏浅",
            "日常应该如何保养和清洁设备",
        ],
        INTENT_AMBIGUOUS: [
            "HAK180和HAK200有什么区别",
            "这两个型号哪个更好用",
            "新旧版本对比怎么样",
            "标准版和专业版的差异在哪里",
        ],
        INTENT_OUT_OF_SCOPE: [
            "你好，你能做什么",
            "陪我聊聊天吧",
            "今天天气怎么样",
            "给我讲个笑话",
            "你是谁，你从哪里来",
        ],
    },
    l2_high=0.75,
    l2_margin=0.05,
    strategies={
        INTENT_EXACT: IntentStrategy(
            dense_weight=0.3, sparse_weight=0.7, ef_search=64,
            per_path_limit=10, hybrid_limit=5,
            enable_hyde=False, enable_web_search=False,
            rerank_max_topk=10, rerank_min_topk=3,
        ),
        INTENT_SEMANTIC: IntentStrategy(
            dense_weight=0.8, sparse_weight=0.2, ef_search=128,
            per_path_limit=10, hybrid_limit=5,
            enable_hyde=True, enable_web_search=True,
            rerank_max_topk=10, rerank_min_topk=3,
        ),
        INTENT_AMBIGUOUS: IntentStrategy(
            dense_weight=0.6, sparse_weight=0.4, ef_search=128,
            per_path_limit=20, hybrid_limit=10,
            enable_hyde=True, enable_web_search=True,
            rerank_max_topk=10, rerank_min_topk=3,
        ),
        INTENT_OUT_OF_SCOPE: IntentStrategy(
            dense_weight=0.8, sparse_weight=0.2, ef_search=0,
            per_path_limit=10, hybrid_limit=5,
            enable_hyde=False, enable_web_search=False,
            rerank_max_topk=10, rerank_min_topk=3,
        ),
        INTENT_DEFAULT: IntentStrategy(
            dense_weight=0.8, sparse_weight=0.2, ef_search=0,
            per_path_limit=10, hybrid_limit=5,
            enable_hyde=True, enable_web_search=True,
            rerank_max_topk=10, rerank_min_topk=3,
        ),
    },
    out_of_scope_answer="抱歉，这个问题超出了本知识库的范围，我主要回答产品参数、使用操作与故障处理相关问题。",
)


def get_strategy_dict(intent: str) -> dict:
    """取意图对应的检索策略字典，未知意图回退默认策略"""
    strategy = intent_config.strategies.get(intent) or intent_config.strategies[INTENT_DEFAULT]
    return strategy.to_dict()
