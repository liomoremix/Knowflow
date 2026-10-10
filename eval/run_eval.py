# eval/run_eval.py

"""召回评测 CLI：在评测集上跑检索并产出指标报告。

用法示例：
    python -m eval.run_eval --tag baseline                        # 现状基线（默认策略）
    python -m eval.run_eval --tag exact --intent EXACT            # 固定按精确查询策略跑
    python -m eval.run_eval --tag ef256 --ef-search 256           # 只调 HNSW ef
    python -m eval.run_eval --tag intent-auto --use-intent        # 走三层漏斗自动判意图
    python -m eval.run_eval --tag recall-layer --no-rerank        # 只测召回层，不测 rerank
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime

from eval.config import SCENARIOS, dataset_path, eval_config
from eval.dataset import load_records
from eval.metrics import evaluate_query, format_report, summarize
from eval.retrieve import (
    EvalRetrievalError,
    resolve_strategy,
    retrieve,
    retrieve_raw,
    strategy_from_query,
    to_ranked_docs,
)
from tool.logger import logger

# 错误率超过这个比例说明是环境问题（Milvus 挂了/模型没加载），指标不可信，直接失败
MAX_ERROR_RATE = 0.10


def git_commit() -> str:
    """当前代码版本，写进结果文件便于回溯「这份指标是哪次提交跑出来的」"""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def load_samples(scenarios: list[str], limit: int = None, version: str = None) -> list[dict]:
    """加载评测样本（已自动剔除人工驳回的记录）"""
    samples = []
    for scenario in scenarios:
        records = load_records(dataset_path(scenario, version))
        if limit:
            records = records[:limit]
        if not records:
            logger.warning(f"[{scenario}] 数据集为空或不存在：{dataset_path(scenario, version)}")
        samples.extend(records)
    return samples


def run(samples: list[dict], args) -> tuple[list[dict], dict]:
    """
    逐条跑检索并打分
    :return: (打分明细列表, 运行统计)
    """
    k_list = eval_config.k_list
    rows = []
    errors = 0
    intent_hits = 0
    intent_total = 0
    latencies = []
    started = time.time()

    for idx, record in enumerate(samples, start=1):
        query = record["query"]

        try:
            reuse = None
            if args.use_intent:
                strategy, intent_state = strategy_from_query(query)
                reuse = intent_state
                predicted = intent_state.get("intent")
                expected = record.get("expected_intent")
                if expected:
                    intent_total += 1
                    intent_hits += int(predicted == expected)
            else:
                strategy = resolve_strategy(
                    intent=args.intent,
                    ef_search=args.ef_search,
                    dense_weight=args.dense_weight,
                    sparse_weight=args.sparse_weight,
                    enable_hyde=None if args.hyde is None else args.hyde,
                )

            item_names = None
            if not args.no_item_filter and record.get("gold_item_name"):
                item_names = [record["gold_item_name"]]

            query_started = time.time()
            fetch = retrieve_raw if args.no_rerank else retrieve
            chunks = fetch(
                query,
                strategy=strategy,
                item_names=item_names,
                collection=args.collection,
            )
            latencies.append(time.time() - query_started)

            docs = to_ranked_docs(chunks)
            scores = evaluate_query(
                (record["gold_file_title"], record["gold_title"], int(record.get("gold_part") or 0)),
                docs,
                k_list,
            )
            rows.append({
                "qid": record.get("qid"),
                "scenario": record["scenario"],
                "query": query,
                "scores": scores,
                "detail": {
                    "rank": scores["rank"],
                    "gold": [record["gold_file_title"], record["gold_title"], int(record.get("gold_part") or 0)],
                    "top": [[d.get("file_title"), d.get("title"), d.get("part")] for d in docs[:max(k_list)]],
                    "retrieved": len(docs),
                },
            })

        except EvalRetrievalError as e:
            errors += 1
            logger.error(f"[{idx}/{len(samples)}] 检索失败: {e}")
        except Exception as e:
            errors += 1
            logger.exception(f"[{idx}/{len(samples)}] 评测异常 query={query!r}: {e}")

        if idx % 20 == 0 or idx == len(samples):
            done = len(rows)
            hit = sum(1 for r in rows if r["scores"]["rank"]) if done else 0
            logger.info(f"进度 {idx}/{len(samples)} | 有效 {done} | 命中 {hit} | 失败 {errors}")

    stats = {
        "total_samples": len(samples),
        "scored": len(rows),
        "errors": errors,
        "error_rate": round(errors / len(samples), 4) if samples else 0.0,
        "elapsed_seconds": round(time.time() - started, 2),
        "latency_avg_ms": round(sum(latencies) / len(latencies) * 1000, 1) if latencies else None,
        "latency_p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95) - 1] * 1000, 1) if latencies else None,
        "intent_total": intent_total,
        "intent_hits": intent_hits,
        "intent_accuracy": round(intent_hits / intent_total, 4) if intent_total else None,
    }
    return rows, stats


def build_result(rows: list[dict], stats: dict, args, scenarios: list[str]) -> dict:
    """组装结果文件内容"""
    return {
        "meta": {
            "tag": args.tag,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_commit(),
            "dataset_version": args.version or eval_config.dataset_version,
            "scenarios": scenarios,
            "collection": args.collection or "(.env CHUNKS_COLLECTION)",
            "k_list": eval_config.k_list,
            "mode": "recall_layer" if args.no_rerank else "full_chain",
            "use_intent": bool(args.use_intent),
            "intent": args.intent,
            "overrides": {
                "ef_search": args.ef_search,
                "dense_weight": args.dense_weight,
                "sparse_weight": args.sparse_weight,
                "hyde": args.hyde,
            },
            "filter_by_item_name": not args.no_item_filter,
            "eval_limits": {
                "per_path_limit": eval_config.eval_per_path_limit,
                "hybrid_limit": eval_config.eval_hybrid_limit,
            },
            **stats,
        },
        **summarize([{"scenario": r["scenario"], "scores": r["scores"]} for r in rows], eval_config.k_list),
        "details": [] if args.no_details else rows,
    }


def result_path(tag: str, out: str = None) -> str:
    """结果文件路径：eval/results/{时间戳}_{tag}.json"""
    if out:
        return out
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_config.result_dir.mkdir(parents=True, exist_ok=True)
    return str(eval_config.result_dir / f"{ts}_{tag}.json")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="跑召回评测并产出指标报告")
    parser.add_argument("--tag", default="run", help="结果文件标签，例如 baseline / ef256")
    parser.add_argument("--scenario", action="append", choices=list(SCENARIOS),
                        help="只评测指定场景，可重复传；默认全部")
    parser.add_argument("--limit", type=int, help="每场景最多评测条数（快速冒烟用）")
    parser.add_argument("--version", help="数据集版本前缀，默认 v1")
    parser.add_argument("--collection", help="检索目标集合，默认 .env 的 CHUNKS_COLLECTION")
    parser.add_argument("--intent", choices=["EXACT", "SEMANTIC", "AMBIGUOUS", "OUT_OF_SCOPE", "DEFAULT"],
                        help="固定按某个意图策略跑；不传则用 DEFAULT（等同改造前行为）")
    parser.add_argument("--use-intent", action="store_true", help="走三层意图漏斗自动判定策略")
    parser.add_argument("--ef-search", type=int, help="覆盖 HNSW ef")
    parser.add_argument("--dense-weight", type=float, help="覆盖稠密权重（稀疏权重自动取补数）")
    parser.add_argument("--sparse-weight", type=float, help="覆盖稀疏权重")
    parser.add_argument("--hyde", type=lambda v: v.lower() in ("1", "true", "yes"), default=None,
                        help="覆盖 HyDE 开关：true/false")
    parser.add_argument("--no-rerank", action="store_true", help="只测召回层（RRF 输出），跳过 rerank")
    parser.add_argument("--no-item-filter", action="store_true", help="不按商品名过滤，全库检索")
    parser.add_argument("--no-details", action="store_true", help="结果文件不写逐条明细（省体积）")
    parser.add_argument("--output", help="结果文件路径，默认 eval/results/{时间戳}_{tag}.json")
    args = parser.parse_args(argv)

    scenarios = args.scenario or list(SCENARIOS)
    samples = load_samples(scenarios, args.limit, args.version)
    if not samples:
        logger.error("没有可用评测样本，请先运行 python -m eval.generate_dataset")
        return 2

    logger.info(f"开始评测：{len(samples)} 条样本 | 场景={scenarios} | 模式={'召回层' if args.no_rerank else '完整链路'}")
    rows, stats = run(samples, args)

    if stats["error_rate"] > MAX_ERROR_RATE:
        logger.error(f"检索失败率 {stats['error_rate']:.1%} 超过 {MAX_ERROR_RATE:.0%}，"
                     f"指标不可信（检查 Milvus / 模型 / rerank 服务是否可用）")
        return 3

    result = build_result(rows, stats, args, scenarios)
    path = result_path(args.tag, args.output)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    metric_keys = [f"recall@{k}" for k in eval_config.k_list] + ["mrr"]
    print("\n" + format_report(result, metric_keys))
    if stats["intent_accuracy"] is not None:
        print(f"\n意图识别准确率: {stats['intent_accuracy']:.2%} ({stats['intent_hits']}/{stats['intent_total']})")
    print(f"\n耗时 {stats['elapsed_seconds']}s | 平均 {stats['latency_avg_ms']}ms | "
          f"p95 {stats['latency_p95_ms']}ms | 失败 {stats['errors']}/{stats['total_samples']}")
    print(f"结果已写入: {path}")
    logger.info(f"评测完成，结果文件: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
