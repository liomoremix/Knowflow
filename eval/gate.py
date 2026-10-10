# eval/gate.py

"""发布门禁 CLI：候选版本的召回率跌幅超过阈值就拦住，不让上线。

对应图中要求「召回率下降超过 2% 时阻止发布」。P3 蓝绿索引切换前强制调用本脚本。

用法：
    python -m eval.gate --baseline eval/results/xxx_baseline.json --candidate eval/results/yyy_ef256.json
    python -m eval.gate --baseline ... --candidate ... --metric recall@5 --max-drop 0.03

退出码：0=通过，1=拦截（召回率跌幅超阈值），2=输入不可比/文件有问题
"""

import argparse
import json
import sys

from eval.config import eval_config
from eval.metrics import compare
from tool.logger import logger


def load_result(path: str) -> dict:
    """读取 run_eval 的结果文件"""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"结果文件不存在: {path}")
        sys.exit(2)
    except json.JSONDecodeError as e:
        logger.error(f"结果文件不是合法 JSON: {path} ({e})")
        sys.exit(2)


def check_comparable(base: dict, cand: dict) -> list[str]:
    """
    两份结果必须可比，否则门禁结论没有意义
    :return: 阻断性问题列表（非空表示不可比）
    """
    problems = []
    bm, cm = base.get("meta", {}), cand.get("meta", {})

    if bm.get("mode") != cm.get("mode"):
        problems.append(f"评测模式不同：baseline={bm.get('mode')} candidate={cm.get('mode')}")
    if bm.get("dataset_version") != cm.get("dataset_version"):
        problems.append(f"数据集版本不同：baseline={bm.get('dataset_version')} candidate={cm.get('dataset_version')}")
    if bm.get("collection") != cm.get("collection"):
        # 蓝绿切换场景下集合名本来就会变，这里只提示不阻断
        logger.warning(f"检索集合不同：baseline={bm.get('collection')} candidate={cm.get('collection')}")
    if bm.get("filter_by_item_name") != cm.get("filter_by_item_name"):
        problems.append("商品名过滤开关不同，召回口径不一致")

    bs, cs = bm.get("scored") or 0, cm.get("scored") or 0
    if bs and cs and abs(bs - cs) / bs > 0.05:
        problems.append(f"有效样本量差异过大：baseline={bs} candidate={cs}")
    if cs == 0:
        problems.append("candidate 没有任何有效样本")

    return problems


def format_gate_table(diff: dict, max_drop: float) -> str:
    """渲染门禁对比表"""
    metric = diff["metric"]
    lines = [
        f"门禁指标: {metric}    允许最大跌幅: {max_drop:.2%}",
        "",
        f"{'范围':<28}{'基线':>10}{'候选':>10}{'变化':>12}{'结论':>8}",
        "-" * 68,
    ]
    for row in diff["rows"]:
        if row["delta"] is None:
            lines.append(f"{row['scope']:<28}{'-':>10}{'-':>10}{'缺数据':>12}{'跳过':>8}")
            continue
        blocked = row["delta"] < -max_drop
        verdict = "拦截" if blocked else ("通过" if row["delta"] >= 0 else "轻微下降")
        sign = "+" if row["delta"] > 0 else ""
        lines.append(
            f"{row['scope']:<28}{row['base']:>10.4f}{row['cand']:>10.4f}"
            f"{sign}{row['delta']:>11.4f}{verdict:>8}"
        )
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="召回率发布门禁")
    parser.add_argument("--baseline", required=True, help="基线结果文件（run_eval 产出）")
    parser.add_argument("--candidate", required=True, help="候选结果文件（run_eval 产出）")
    parser.add_argument("--metric", default=None, help=f"门禁指标，默认 {eval_config.gate_metric}")
    parser.add_argument("--max-drop", type=float, default=None,
                        help=f"允许的最大绝对跌幅，默认 {eval_config.gate_recall_drop}")
    parser.add_argument("--force", action="store_true", help="输入不可比时仍然继续（只警告）")
    args = parser.parse_args(argv)

    metric = args.metric or eval_config.gate_metric
    max_drop = eval_config.gate_recall_drop if args.max_drop is None else args.max_drop

    base = load_result(args.baseline)
    cand = load_result(args.candidate)

    problems = check_comparable(base, cand)
    if problems:
        for problem in problems:
            logger.error(f"两份结果不可比: {problem}")
        if not args.force:
            logger.error("门禁无法判断，退出码 2（加 --force 可强行比较）")
            return 2
        logger.warning("--force 已指定，继续比较，但结论仅供参考")

    diff = compare(base, cand, metric)
    print()
    print(format_gate_table(diff, max_drop))
    print()

    if diff["worst_delta"] is None:
        logger.error(f"基线与候选都缺少 {metric} 数据，无法判断")
        return 2

    if diff["worst_delta"] < -max_drop:
        logger.error(
            f"门禁拦截：{diff['worst_scope']} 的 {metric} 下降 {abs(diff['worst_delta']):.4f} "
            f"> 阈值 {max_drop:.4f}，禁止发布/切换索引"
        )
        print(f"结果: BLOCKED  ({diff['worst_scope']} {metric} {diff['worst_delta']:+.4f})")
        return 1

    logger.info(f"门禁通过：最大跌幅 {abs(min(diff['worst_delta'], 0)):.4f} <= 阈值 {max_drop:.4f}")
    print(f"结果: PASSED  (最差 {diff['worst_scope']} {metric} {diff['worst_delta']:+.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
