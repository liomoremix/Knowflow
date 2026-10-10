# eval/sample_review.py

"""人工抽查 CLI：LLM 出的题不一定靠谱，按比例抽出来给人核对，核对结果回写数据集。

对应图中要求「10% 人工抽查」。被判为驳回(verified=False)的样本在 run_eval 时自动剔除。

用法：
    python -m eval.sample_review export                       # 全部场景抽 10% 导出待审文件
    python -m eval.sample_review export --scenario troubleshoot --ratio 0.2
    python -m eval.sample_review apply --review eval/review/xxx.jsonl   # 把人工结论写回数据集
    python -m eval.sample_review status                       # 查看各场景审核进度
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

from eval.config import SCENARIOS, dataset_path, eval_config
from eval.dataset import load_records, rewrite_records
from tool.logger import logger

REVIEW_DIR = eval_config.dataset_dir.parent / "eval" / "review"

# 待审文件字段：verdict 由人工填 "ok" / "reject"
REVIEW_FIELDS = [
    "qid", "scenario", "query", "expected_intent",
    "gold_file_title", "gold_title", "gold_part", "gold_item_name",
    "gold_content", "gold_content_hash", "verdict", "note",
]


def review_path(scenario: str = "all") -> Path:
    """待审文件路径"""
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    return REVIEW_DIR / f"review_{scenario}_{ts}.jsonl"


def export(scenarios: list[str], ratio: float, seed: int = None, out: str = None) -> int:
    """
    抽样导出待审文件
    :param scenarios: 场景列表
    :param ratio: 抽查比例
    :param seed: 随机种子
    :param out: 输出路径，默认自动生成
    :return: 导出条数
    """
    rng = random.Random(seed)
    path = Path(out) if out else review_path("all" if len(scenarios) > 1 else scenarios[0])
    path.parent.mkdir(parents=True, exist_ok=True)

    exported = 0
    with path.open("w", encoding="utf-8") as f:
        for scenario in scenarios:
            records = load_records(dataset_path(scenario), include_rejected=True)
            # 只抽还没审过的，避免重复劳动
            pending = [r for r in records if r.get("verified") is None]
            count = max(1, round(len(pending) * ratio)) if pending else 0
            picked = rng.sample(pending, min(count, len(pending))) if pending else []

            for record in picked:
                row = {
                    "qid": record.get("qid"),
                    "scenario": scenario,
                    "query": record.get("query"),
                    "expected_intent": record.get("expected_intent"),
                    "gold_file_title": record.get("gold_file_title"),
                    "gold_title": record.get("gold_title"),
                    "gold_part": record.get("gold_part"),
                    "gold_item_name": record.get("gold_item_name"),
                    "gold_content": record.get("gold_content"),
                    "gold_content_hash": record.get("gold_content_hash"),
                    # 人工填写：ok = 问题合理且答案确实在资料里；reject = 出题质量差
                    "verdict": "",
                    "note": "",
                }
                f.write(json.dumps({k: row.get(k) for k in REVIEW_FIELDS}, ensure_ascii=False) + "\n")
                exported += 1

            logger.info(f"[{SCENARIOS[scenario].label}] 待审 {len(pending)} 条，抽出 {len(picked)} 条")

    logger.info(f"已导出 {exported} 条待审样本 → {path}")
    print(f"\n请编辑该文件，把每行的 verdict 填成 ok 或 reject，然后执行：")
    print(f"  python -m eval.sample_review apply --review {path}")
    return exported


def apply_verdicts(review_file: str) -> int:
    """
    把人工结论回写数据集的 verified 字段
    :param review_file: 待审文件路径
    :return: 实际更新的样本数
    """
    path = Path(review_file)
    if not path.exists():
        logger.error(f"待审文件不存在: {review_file}")
        return 0

    verdicts = {}
    notes = {}
    skipped = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            verdict = (row.get("verdict") or "").strip().lower()
            if verdict in ("ok", "y", "yes", "true", "1", "通过"):
                verdicts[row["qid"]] = True
            elif verdict in ("reject", "rejected", "no", "n", "false", "0", "驳回"):
                verdicts[row["qid"]] = False
            else:
                skipped += 1
                continue
            notes[row["qid"]] = row.get("note") or ""

    if skipped:
        logger.warning(f"{skipped} 条 verdict 为空或无法识别，保持未审核状态")
    if not verdicts:
        logger.error("没有任何可回写的审核结论")
        return 0

    updated = 0
    for scenario in SCENARIOS:
        ds_path = dataset_path(scenario)
        records = load_records(ds_path, include_rejected=True)
        changed = False
        for record in records:
            qid = record.get("qid")
            if qid in verdicts and record.get("verified") is not verdicts[qid]:
                record["verified"] = verdicts[qid]
                record["review_note"] = notes.get(qid, "")
                changed = True
                updated += 1
        if changed:
            rewrite_records(ds_path, records)
            logger.info(f"[{SCENARIOS[scenario].label}] 回写完成 → {ds_path}")

    ok = sum(1 for v in verdicts.values() if v)
    logger.info(f"共更新 {updated} 条：确认 {ok}，驳回 {len(verdicts) - ok}（驳回样本不再计入评测）")
    return updated


def status(scenarios: list[str]) -> int:
    """打印各场景审核进度"""
    print(f"\n{'场景':<14}{'总数':>8}{'已确认':>8}{'已驳回':>8}{'未审核':>8}{'审核率':>10}")
    print("-" * 58)
    total = verified = rejected = pending = 0
    for scenario in scenarios:
        records = load_records(dataset_path(scenario), include_rejected=True)
        v = sum(1 for r in records if r.get("verified") is True)
        r_ = sum(1 for r in records if r.get("verified") is False)
        p = sum(1 for r in records if r.get("verified") is None)
        rate = (v + r_) / len(records) if records else 0.0
        print(f"{SCENARIOS[scenario].label:<14}{len(records):>8}{v:>8}{r_:>8}{p:>8}{rate:>10.1%}")
        total += len(records)
        verified += v
        rejected += r_
        pending += p
    print("-" * 58)
    rate = (verified + rejected) / total if total else 0.0
    print(f"{'合计':<14}{total:>8}{verified:>8}{rejected:>8}{pending:>8}{rate:>10.1%}")
    print(f"\n目标抽查比例: {eval_config.sample_review_ratio:.0%}（当前 {rate:.1%}）")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="评测集人工抽查")
    sub = parser.add_subparsers(dest="command", required=True)

    p_export = sub.add_parser("export", help="抽样导出待审文件")
    p_export.add_argument("--scenario", action="append", choices=list(SCENARIOS), help="指定场景，可重复传")
    p_export.add_argument("--ratio", type=float, default=None, help=f"抽查比例，默认 {eval_config.sample_review_ratio}")
    p_export.add_argument("--seed", type=int, help="随机种子")
    p_export.add_argument("--out", help="输出文件路径")

    p_apply = sub.add_parser("apply", help="把人工结论回写数据集")
    p_apply.add_argument("--review", required=True, help="待审文件路径")

    p_status = sub.add_parser("status", help="查看审核进度")
    p_status.add_argument("--scenario", action="append", choices=list(SCENARIOS), help="指定场景，可重复传")

    args = parser.parse_args(argv)
    scenarios = (args.scenario or list(SCENARIOS)) if hasattr(args, "scenario") else list(SCENARIOS)

    if args.command == "export":
        ratio = eval_config.sample_review_ratio if args.ratio is None else args.ratio
        export(scenarios, ratio, args.seed, args.out)
        return 0
    if args.command == "apply":
        return 0 if apply_verdicts(args.review) else 1
    return status(scenarios)


if __name__ == "__main__":
    sys.exit(main())
