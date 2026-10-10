# processor/query_processor/nodes/node_intent_recognition.py

import json
import re
from typing import Dict, List, Optional, Tuple

from langchain_core.messages import HumanMessage, SystemMessage

from config.intent_config import (
    INTENT_AMBIGUOUS,
    INTENT_DEFAULT,
    INTENT_EXACT,
    INTENT_OUT_OF_SCOPE,
    INTENT_SEMANTIC,
    get_strategy_dict,
    intent_config,
)
from processor.query_processor.base import NodeBase
from processor.query_processor.prompt.intent_recognition import (
    INTENT_L3_SYSTEM_PROMPT,
    INTENT_L3_TEMPLATE,
)
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.json_format_utils import serialize_json
from utils.llm_utils import get_llm_client

VALID_INTENTS = {INTENT_EXACT, INTENT_SEMANTIC, INTENT_AMBIGUOUS, INTENT_OUT_OF_SCOPE, INTENT_DEFAULT}

# 意图原型向量缓存：类别 -> 单位向量列表（首次 L2 使用时计算，避免 import 期加载模型）
_prototype_vector_cache: Dict[str, List[list]] = {}


def _unit_vec(vec: list) -> list:
    """向量归一化，模为 0 时原样返回"""
    norm = sum(x * x for x in vec) ** 0.5
    if norm == 0.0:
        return vec
    return [x / norm for x in vec]


def _get_prototype_vectors() -> Dict[str, List[list]]:
    """获取各类意图原型句的单位向量（模块级缓存，只算一次）"""
    global _prototype_vector_cache
    if _prototype_vector_cache:
        return _prototype_vector_cache
    for category, sentences in intent_config.l2_prototypes.items():
        dense_list = generate_embeddings(sentences).get("dense")
        _prototype_vector_cache[category] = [_unit_vec(v) for v in dense_list]
    return _prototype_vector_cache


class NodeIntentRecognition(NodeBase):
    """
    节点功能：意图识别三层漏斗（L1 规则 → L2 原型向量 → L3 LLM 兜底），
    识别结果驱动下游检索策略（dense/sparse 权重、HyDE/联网开关、ef 与 topk）。
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_intent_recognition"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        必要参数：rewritten_query 或 original_query
        更新参数：intent、intent_confidence、intent_layer、strategy、query_dense_vec、query_sparse_vec、answer(超范围时)

        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """

        # 上游已产生直答（确认反问/拒答）时直接透传
        if state.get("answer"):
            return state

        query = state.get("rewritten_query") or state.get("original_query") or ""
        if not query:
            return state

        # 总开关关闭：直通默认策略，行为等同改造前
        if not intent_config.intent_enabled:
            logger.info("意图识别总开关关闭，直通默认策略")
            return self._build_result(state, INTENT_DEFAULT, 1.0, "disabled", None, None)

        # 外部指定意图：跳过识别直接取对应策略
        override = state.get("intent_override")
        if override:
            logger.info(f"使用外部指定意图: {override}")
            return self._build_result(state, override, 1.0, "override", None, None)

        # L1 规则层：关键词/正则，毫秒级零成本
        intent, conf, layer = self._step_1_l1_rules(query)

        # 查询向量在此只算一次，下游检索节点复用（超范围短路无需算）
        dense_vec = sparse_vec = None
        if intent != INTENT_OUT_OF_SCOPE:
            dense_vec, sparse_vec = self._step_2_embed(query)

        if intent is None:
            # L2 轻量模型层：与意图原型向量余弦相似度
            intent_l2, conf_l2, hint = self._step_3_l2_prototype(dense_vec)
            if intent_l2:
                intent, conf, layer = intent_l2, conf_l2, "L2"
            else:
                # L3 LLM 兜底：仅低置信度触发
                intent, conf = self._step_4_l3_llm(query, state.get("history") or [], hint)
                layer = "L3"

        logger.info(f"意图识别结果: intent={intent}, confidence={conf:.3f}, layer={layer}")
        return self._build_result(state, intent, conf, layer, dense_vec, sparse_vec)



    def _step_1_l1_rules(self, query: str) -> Tuple[Optional[str], float, str]:
        """
        L1 规则层：关键词子串 + 型号正则命中即定类（置信度 1.0）
        优先级：超范围(无型号前提) → 歧义 → 语义 → 精确(关键词或型号正则)
        :return: (意图类别或None, 置信度, 层级)
        """
        has_model = any(
            re.search(pattern, query, flags=re.IGNORECASE)
            for pattern in intent_config.l1_patterns.get(INTENT_EXACT, [])
        )

        if not has_model and any(k in query for k in intent_config.l1_keywords.get(INTENT_OUT_OF_SCOPE, [])):
            return INTENT_OUT_OF_SCOPE, 1.0, "L1"
        if any(k in query for k in intent_config.l1_keywords.get(INTENT_AMBIGUOUS, [])):
            return INTENT_AMBIGUOUS, 1.0, "L1"
        if any(k in query for k in intent_config.l1_keywords.get(INTENT_SEMANTIC, [])):
            return INTENT_SEMANTIC, 1.0, "L1"
        if has_model or any(k in query for k in intent_config.l1_keywords.get(INTENT_EXACT, [])):
            return INTENT_EXACT, 1.0, "L1"

        return None, 0.0, "L1"



    def _step_2_embed(self, query: str) -> Tuple[list, dict]:
        """查询向量化（Dense + Sparse），全链路只算这一次"""
        embeddings = generate_embeddings([query])
        return embeddings.get("dense")[0], embeddings.get("sparse")[0]



    def _step_3_l2_prototype(self, dense_vec: list) -> Tuple[Optional[str], float, Tuple[str, float]]:
        """
        L2 轻量模型层：查询向量与各类意图原型向量取最大余弦相似度
        最高分 >= l2_high 且与次高分差 >= l2_margin 时直接定类，否则交给 L3
        :return: (定类意图或None, 最高分, (最高分类别, 最高分)提示元组)
        """
        query_unit = _unit_vec(dense_vec)
        scores = []
        for category, prototype_vecs in _get_prototype_vectors().items():
            score = max(
                (sum(a * b for a, b in zip(query_unit, prototype)) for prototype in prototype_vecs),
                default=0.0
            )
            scores.append((category, score))
        scores.sort(key=lambda item: item[1], reverse=True)

        cat1, s1 = scores[0]
        s2 = scores[1][1] if len(scores) > 1 else 0.0
        logger.info(f"L2 原型相似度: {cat1}={s1:.3f}, 与次高分差={s1 - s2:.3f}")

        if s1 >= intent_config.l2_high and (s1 - s2) >= intent_config.l2_margin:
            return cat1, s1, (cat1, s1)
        return None, s1, (cat1, s1)



    def _step_4_l3_llm(self, query: str, history: list, hint: Tuple[str, float]) -> Tuple[str, float]:
        """L3 兜底层：LLM 意图分类（JSON 输出），失败回退语义意图保全文检索"""
        history_text = "".join(f"{msg.get('role')}: {msg.get('text')}\n" for msg in history[-5:])
        user_prompt = INTENT_L3_TEMPLATE.format(
            history_text=history_text or "（无）",
            query=query,
            hint_intent=hint[0] or "（无）",
            hint_score=round(hint[1], 4),
        )

        try:
            llm = get_llm_client(json_mode=True)
            content = llm.invoke([
                SystemMessage(content=INTENT_L3_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt),
            ]).content

            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "")
            parsed = json.loads(content)

            intent = parsed.get("intent")
            if intent not in VALID_INTENTS:
                logger.error(f"L3 返回非法意图类别: {intent}")
                return INTENT_SEMANTIC, 0.5
            return intent, float(parsed.get("confidence", 0.5))

        except Exception as e:
            logger.error(f"L3 意图识别调用失败: {e}")
            return INTENT_SEMANTIC, 0.5



    def _build_result(self, state: QueryGraphState, intent: str, conf: float, layer: str,
                      dense_vec: Optional[list], sparse_vec: Optional[dict]) -> dict:
        """组装意图识别结果：策略字典 + 查询向量复用 + 超范围短路直答 + top_k 覆盖"""
        if intent not in VALID_INTENTS:
            intent = INTENT_DEFAULT
        strategy = get_strategy_dict(intent)

        top_k = state.get("top_k_override")
        if top_k:
            strategy["hybrid_limit"] = top_k
            strategy["per_path_limit"] = top_k * 2
            strategy["rerank_max_topk"] = top_k

        result = {
            "intent": intent,
            "intent_confidence": conf,
            "intent_layer": layer,
            "strategy": strategy,
        }
        if dense_vec:
            result["query_dense_vec"] = dense_vec
            result["query_sparse_vec"] = sparse_vec
        if intent == INTENT_OUT_OF_SCOPE:
            result["answer"] = intent_config.out_of_scope_answer
        return result




if __name__ == "__main__":

    test_queries = [
        "HAK180烫金机的技术参数是多少",  # 期望 EXACT
        "烫金机应该如何调节转印温度",  # 期望 SEMANTIC
        "HAK180和HAK200有什么区别",  # 期望 AMBIGUOUS
        "你好，你能做什么",  # 期望 OUT_OF_SCOPE
    ]

    node_intent_recognition = NodeIntentRecognition()
    for test_query in test_queries:
        result = node_intent_recognition({"original_query": test_query, "rewritten_query": test_query})
        logger.info(serialize_json({
            "query": test_query,
            "intent": result.get("intent"),
            "intent_confidence": result.get("intent_confidence"),
            "intent_layer": result.get("intent_layer"),
            "strategy": result.get("strategy"),
            "answer": result.get("answer", ""),
        }, indent=4))
