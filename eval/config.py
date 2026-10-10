# eval/config.py

from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# 项目根目录（eval/ 的上一级）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "datasets"
RESULT_DIR = PROJECT_ROOT / "eval" / "results"


# 五个核心评测场景：覆盖精确查询、操作指导、故障排查、对比歧义、长文综合理解
SCENARIO_PARAM_LOOKUP = "param_lookup"
SCENARIO_OPERATION_GUIDE = "operation_guide"
SCENARIO_TROUBLESHOOT = "troubleshoot"
SCENARIO_COMPARISON = "comparison"
SCENARIO_LONG_DOC_QA = "long_doc_qa"


@dataclass
class ScenarioSpec:
    """单个评测场景的生成规格"""
    name: str  # 场景名，同时作为数据集文件名后缀
    label: str  # 中文场景名，用于报告展示
    expected_intent: str  # 该场景期望被识别成的意图类别（用于顺带校验意图准确率）
    min_content_len: int  # 抽样切片的正文长度下限
    max_content_len: int  # 抽样切片的正文长度上限
    instruction: str  # 喂给 LLM 的出题要求


SCENARIOS: dict[str, ScenarioSpec] = {
    SCENARIO_PARAM_LOOKUP: ScenarioSpec(
        name=SCENARIO_PARAM_LOOKUP,
        label="参数查询",
        expected_intent="EXACT",
        min_content_len=50,
        max_content_len=100000,
        instruction=(
            "出一个「查具体参数/规格/型号/数值」的问题，例如额定电压、功率、尺寸、重量、温度范围、配件编号。"
            "问题里不要照抄原文的句子，要像真实用户那样简短地问。"
        ),
    ),
    SCENARIO_OPERATION_GUIDE: ScenarioSpec(
        name=SCENARIO_OPERATION_GUIDE,
        label="操作步骤",
        expected_intent="SEMANTIC",
        min_content_len=200,
        max_content_len=100000,
        instruction=(
            "出一个「怎么做/如何操作/步骤是什么」的问题，答案就在这段资料里。"
            "用口语化的问法，不要出现原文里的小标题原词。"
        ),
    ),
    SCENARIO_TROUBLESHOOT: ScenarioSpec(
        name=SCENARIO_TROUBLESHOOT,
        label="故障排查",
        expected_intent="SEMANTIC",
        min_content_len=200,
        max_content_len=100000,
        instruction=(
            "出一个「出了某种异常现象该怎么办」的排障问题，描述用户能观察到的现象（报错、异响、效果不对），"
            "而不是直接说出原文里的原因或对策名词。"
        ),
    ),
    SCENARIO_COMPARISON: ScenarioSpec(
        name=SCENARIO_COMPARISON,
        label="对比歧义",
        expected_intent="AMBIGUOUS",
        min_content_len=100,
        max_content_len=100000,
        instruction=(
            "出一个「两种做法/两个型号/两个选项怎么选、有什么区别」的对比问题。"
            "如果这段资料只讲了一个对象，就针对该对象的两种设置或两种状态提问。"
        ),
    ),
    SCENARIO_LONG_DOC_QA: ScenarioSpec(
        name=SCENARIO_LONG_DOC_QA,
        label="长文综合",
        expected_intent="SEMANTIC",
        min_content_len=1500,
        max_content_len=100000,
        instruction=(
            "出一个需要通读整段资料才能回答的综合性问题（例如「这部分主要讲了什么注意事项」"
            "「整个过程有哪些关键约束」），答案分散在多处而不是某一句话。"
        ),
    ),
}


@dataclass
class EvalConfig:
    """召回评测体系配置"""
    # ---- 指标 ----
    k_list: list = field(default_factory=lambda: [1, 3, 5, 10])  # recall@k / precision@k / ndcg@k 的 k 取值
    gate_metric: str = "recall@10"  # 门禁看的主指标
    gate_recall_drop: float = 0.02  # 允许的最大绝对跌幅（2%），超过即拦截发布

    # ---- 数据集生成 ----
    quota_per_scenario: int = 200  # 每个场景的目标条数（图中要求 5 场景 × 200）
    dedup_cosine_threshold: float = 0.95  # 生成问题去重阈值：与已有问题余弦相似度高于此值则丢弃
    sample_batch_size: int = 500  # 从 Milvus 一次抽样多少条切片
    gen_llm_model: str = ""  # 出题用的模型，空串表示用 .env 的 LLM_DEFAULT_MODEL

    # ---- 检索复现 ----
    eval_per_path_limit: int = 20  # 评测时单路召回数量（比线上宽，保证 recall@10 有意义）
    eval_hybrid_limit: int = 10  # 评测时混合检索返回数量
    filter_by_item_name: bool = True  # 是否像线上一样按 gold 切片的商品名过滤

    # ---- 人工抽查 ----
    sample_review_ratio: float = 0.10  # 人工抽查比例（图中要求 10%）

    # ---- 线上抽样评测（P5）----
    # 离线评测集衡量的是「对理想问题」的表现，线上抽样衡量的是「对真实用户问题」的表现，
    # 两者差距变大本身就是信号，所以两套指标都留着，靠 mode 标签区分
    online_sample: int = 30  # 每轮抽多少条真实查询（30 条 × 每条 1 次 LLM 调用，成本可控）
    online_judge_top: int = 3  # 每条查询评前几个切片（当时真正交给大模型的那几个）
    online_since_hours: float = 24.0  # 只从最近多少小时的日志里抽，太老的查询不代表当前索引
    online_content_chars: int = 1200  # 回查正文时每条截断到多少字，够裁判判断即可
    online_relevance_min: float = 0.70  # 相关率（前 3 条里至少一条相关）低于此值算异常
    online_low_streak: int = 2  # 连续几轮低于阈值才触发重建判定（单轮抽样抖动大）
    online_judge_model: str = ""  # 裁判模型，空串表示用 .env 的 LLM_DEFAULT_MODEL
    online_exclude_intents: list = field(default_factory=lambda: ["OUT_OF_SCOPE"])  # 闲聊没检索，无从评起

    # ---- 路径 ----
    dataset_dir: Path = DATASET_DIR
    result_dir: Path = RESULT_DIR
    dataset_version: str = "v1"  # 数据集文件名前缀：v1_param_lookup.jsonl


eval_config = EvalConfig()


def dataset_path(scenario: str, version: str = None) -> Path:
    """场景数据集文件路径：datasets/{version}_{scenario}.jsonl"""
    version = version or eval_config.dataset_version
    return eval_config.dataset_dir / f"{version}_{scenario}.jsonl"


def all_dataset_paths(version: str = None) -> list[Path]:
    """全部场景数据集路径（只返回已存在的）"""
    return [p for p in (dataset_path(s, version) for s in SCENARIOS) if p.exists()]
