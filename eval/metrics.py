# eval/metrics.py

"""召回质量指标：recall@k / precision@k / MRR / NDCG@k。

每条评测样本只有一个标准答案（gold 切片），因此：
- recall@k 等于 hit@k（命中即 1，否则 0）
- precision@k 命中时为 1/k，否则 0
- NDCG@k 采用二元相关性，命中时为 1/log2(rank+1)
命中判定用业务三元组 (file_title, title, part)，不用会随重建变化的 chunk_id。
"""

import math
from collections import defaultdict

from eval.dataset import gold_key, hit_key


def rank_of_gold(gold: tuple, retrieved: list[dict]) -> int:
    """gold 在检索结果中的排名（1 起）；未命中返回 0"""
    for idx, doc in enumerate(retrieved, start=1):
        if hit_key(doc) == gold:
            return idx
    return 0


def evaluate_query(gold: tuple, retrieved: list[dict], k_list: list[int]) -> dict:
    """
    计算单条查询的各项指标
    :param gold: 标准答案三元组
    :param retrieved: 按相关性降序的检索结果（元素需含 file_title/title/part）
    :param k_list: 需要计算的 k 值列表
    :return: {"rank": int, "recall@k": ..., "precision@k": ..., "ndcg@k": ..., "mrr": float}
    """
    rank = rank_of_gold(gold, retrieved)
    scores = {"rank": rank, "mrr": (1.0 / rank) if rank else 0.0}

    for k in k_list:
        hit_in_k = bool(rank) and rank <= k
        scores[f"recall@{k}"] = 1.0 if hit_in_k else 0.0
        scores[f"precision@{k}"] = (1.0 / k) if hit_in_k else 0.0
        scores[f"ndcg@{k}"] = (1.0 / math.log2(rank + 1)) if hit_in_k else 0.0

    return scores


METRIC_NAMES = ["recall", "precision", "ndcg"]


def aggregate(rows: list[dict], k_list: list[int]) -> dict:
    """
    对一批单查询指标求均值
    :param rows: evaluate_query 的结果列表
    :param k_list: k 值列表
    :return: {"count": n, "recall@k": ..., "precision@k": ..., "ndcg@k": ..., "mrr": ...}
    """
    if not rows:
        return {"count": 0}

    total = {"count": len(rows), "mrr": _mean(r["mrr"] for r in rows)}
    for name in METRIC_NAMES:
        for k in k_list:
            key = f"{name}@{k}"
            total[key] = _mean(r[key] for r in rows)
    return total


def _mean(values) -> float:
    values = list(values)
    return round(sum(values) / len(values), 6) if values else 0.0


def summarize(samples: list[dict], k_list: list[int]) -> dict:
    """
    按场景聚合 + 计算总均值
    :param samples: [{"scenario": str, "scores": evaluate_query 结果}, ...]
    :return: {"by_scenario": {场景: 指标}, "overall": 指标, "macro_overall": 场景均值}
    """
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["scenario"]].append(sample["scores"])

    by_scenario = {scenario: aggregate(rows, k_list) for scenario, rows in sorted(grouped.items())}
    all_rows = [s["scores"] for s in samples]

    return {
        "by_scenario": by_scenario,
        # overall：所有查询一起算（样本量大的场景权重更高）
        "overall": aggregate(all_rows, k_list),
        # macro_overall：各场景指标再取平均（每个场景等权，避免长尾场景被淹没）
        "macro_overall": _macro_average(list(by_scenario.values()), k_list),
    }


def _macro_average(scenario_metrics: list[dict], k_list: list[int]) -> dict:
    """各场景等权的宏平均"""
    valid = [m for m in scenario_metrics if m.get("count")]
    if not valid:
        return {"count": 0}

    out = {"count": sum(m["count"] for m in valid), "scenarios": len(valid)}
    out["mrr"] = _mean(m["mrr"] for m in valid)
    for name in METRIC_NAMES:
        for k in k_list:
            key = f"{name}@{k}"
            out[key] = _mean(m[key] for m in valid)
    return out


def compare(baseline: dict, candidate: dict, metric: str) -> dict:
    """
    对比两份评测结果的同一个指标，供门禁判断
    :param baseline: 基线结果（run_eval 输出文件的 "by_scenario"+"overall" 部分）
    :param candidate: 候选结果
    :param metric: 指标名，如 "recall@10"
    :return: {"rows": [{scope, base, cand, delta}], "worst_delta": float, "worst_scope": str}
    """
    rows = []

    scopes = sorted(set(baseline.get("by_scenario", {})) | set(candidate.get("by_scenario", {})))
    for scope in scopes:
        base = baseline.get("by_scenario", {}).get(scope, {})
        cand = candidate.get("by_scenario", {}).get(scope, {})
        rows.append(_delta_row(f"scenario:{scope}", base.get(metric), cand.get(metric)))

    for scope in ("overall", "macro_overall"):
        rows.append(_delta_row(scope, baseline.get(scope, {}).get(metric), candidate.get(scope, {}).get(metric)))

    measured = [r for r in rows if r["delta"] is not None]
    worst = min(measured, key=lambda r: r["delta"]) if measured else {"delta": None, "scope": "-"}
    return {"metric": metric, "rows": rows, "worst_delta": worst["delta"], "worst_scope": worst["scope"]}


def _delta_row(scope: str, base, cand) -> dict:
    """单个对比行；任一侧缺数据时 delta 记为 None（无法判断，不作为拦截依据）"""
    if base is None or cand is None:
        return {"scope": scope, "base": base, "cand": cand, "delta": None}
    return {"scope": scope, "base": round(base, 6), "cand": round(cand, 6), "delta": round(cand - base, 6)}


def format_report(summary: dict, metric_keys: list[str]) -> str:
    """把聚合结果渲染成对齐的文本表格，便于终端查看"""
    scopes = list(summary["by_scenario"].items())
    scopes.append(("OVERALL(样本加权)", summary["overall"]))
    scopes.append(("OVERALL(场景等权)", summary["macro_overall"]))

    width = max(len(str(name)) for name, _ in scopes) + 2
    header = "场景".ljust(width) + "".join(f"{m:>14}" for m in metric_keys) + f"{'count':>8}"
    lines = [header, "-" * len(header)]
    for name, metrics in scopes:
        if not metrics.get("count"):
            continue
        cells = "".join(f"{metrics.get(m, 0.0):>14.4f}" for m in metric_keys)
        lines.append(f"{str(name).ljust(width)}{cells}{metrics['count']:>8}")
    return "\n".join(lines)


def gold_of(record: dict) -> tuple:
    """数据集记录 → 标准答案三元组（对外暴露，避免调用方直接依赖 dataset 内部）"""
    return gold_key(record)
