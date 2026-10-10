# processor/index_manager/scheduler.py

"""索引巡检调度器：定期采集信号 → 判定索引是否劣化 → （可选）自动触发一次带门禁的重建。

它解决的问题
------------
P3/P4 已经把「信号」都采出来了（sealed segment、行数漂移、索引类型、p99、慢查询率、
线上抽样相关率），但信号本身不会动。没有这一层，就得靠人每天去看板扫一眼——
而索引劣化恰恰是那种「不看不疼、发现时已经很疼」的问题。

判定分成两类，处理方式完全不同
------------------------------
· **重建能解决的**（碎片化、索引参数没吃到、线上相关率连续下滑、p99 劣化）
  → 触发一次蓝绿重建。重建走的是 P3 那套完整流程：新集合 + 双写 + 一致性校验 +
  召回门禁，门禁不过就不切，所以「自动触发」本身并不危险。
· **重建解决不了的**（行数与注册表对不上、空检索占比高）
  → 只告警。行数漂移的根因是导入漏写或双写失败，重建只是把错误的数据搬一遍；
  空检索多半是知识库根本没这份资料，属于内容问题不是索引问题。
  把这两类混在一起自动重建，会掩盖真正该修的东西。

安全边界
--------
自动触发的重建默认停在 ready（scheduler_prepare_only=True）：建新集合、双写、校验、
门禁都自动做，但**切别名这一步留给人**。流量切换是整个流程里唯一不可逆且立刻影响
所有用户的动作，也是线上事故里最贵的那一类。确认没问题后一条命令即可上线：
    python -m processor.index_manager.rebuild --switch --version <N> --yes
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from pymongo import DESCENDING

from config.index_config import index_config
from config.monitor_config import monitor_config
from eval.config import eval_config
from tool.logger import logger
from utils import index_registry_utils as registry
from utils.mongo_ops_utils import as_dict, get_collection
from utils.query_log_utils import stats as query_log_stats

COLLECTION_NAME = "scheduler_runs"
INDEXES = [
    ([("ts", DESCENDING)], {}),
]

# 一轮巡检的默认时间窗：与调度周期一致，避免每轮都全量扫 query_log
_LOOKBACK_HOURS_DEFAULT = 24.0


# ==================== 采集信号 ====================

def collect_quality() -> Dict[str, Any]:
    """索引侧信号：sealed segment / 行数漂移 / 索引类型（P3 的代理指标）"""
    from processor.index_manager.quality import index_quality

    try:
        quality = index_quality()
    except Exception as e:
        logger.error(f"采集索引质量指标失败: {e}")
        return {"available": False, "error": str(e)[:200]}

    segments = quality.get("segments") or {}
    dense = quality.get("dense_index") or {}
    rows = quality.get("rows")
    warnings = quality.get("warnings") or []
    # rows == -1 是「根本没量到」（Milvus 连不上），不是「量到 0 行」。此时所有子字段都是 None，
    # decide 会一路跳过然后报 healthy——把「监控瞎了」当成「索引健康」是最坏的一种误报
    measured = rows is not None and rows >= 0
    return {
        "available": measured,
        # 没量到时把原因带上，否则告警文案只剩「未知原因」，值班还得自己去翻日志
        "error": None if measured else (warnings[0] if warnings else "Milvus 未返回行数"),
        "collection": quality.get("collection"),
        "version": quality.get("version"),
        "rows": rows,
        "registry_rows": quality.get("registry_rows"),
        "sealed_segments": segments.get("sealed"),
        "growing_segments": segments.get("growing"),
        "dense_index_type": dense.get("index_type") or dense.get("index"),
        "warnings": warnings,
    }


def collect_latency(since: Optional[float] = None) -> Dict[str, Any]:
    """
    延迟侧信号：从 query_log 聚合出 p99 与慢查询率
    p99 取各意图里的最大值：分组算不出真正的全局 p99（要合并分位数得留原始样本），
    取最大值偏保守，用于「是否劣化」的判定足够，不会漏报只会略微早报
    """
    try:
        summary = query_log_stats(since=since) or {}
    except Exception as e:
        logger.error(f"读取 query_log 统计失败: {e}")
        return {"available": False, "error": str(e)[:200]}

    by_intent = summary.get("by_intent") or {}
    p99 = max((v.get("p99_latency_ms") or 0.0) for v in by_intent.values()) if by_intent else 0.0
    return {
        "available": bool(summary),
        "total": summary.get("total", 0),
        "slow_total": summary.get("slow_total", 0),
        "slow_rate": summary.get("slow_rate", 0.0),
        "p99_latency_ms": round(p99, 1),
        "by_intent": {k: {"count": v.get("count"), "p99_latency_ms": v.get("p99_latency_ms")}
                      for k, v in by_intent.items()},
    }


def collect_online() -> Dict[str, Any]:
    """线上抽样信号：最近一轮相关率 + 连续低于阈值的轮数"""
    from eval.online_eval import low_streak, recent_runs

    runs = recent_runs(limit=eval_config.online_low_streak + 3)
    latest = runs[0] if runs else {}
    return {
        "available": bool(runs),
        "runs": len(runs),
        "run_id": latest.get("run_id"),
        "sampled": latest.get("sampled"),
        "recall@1": latest.get("recall@1"),
        "recall@3": latest.get("recall@3"),
        "useful_rate": latest.get("useful_rate"),
        "judge_rate": latest.get("judge_rate"),
        "empty_retrieval": (latest.get("by_status") or {}).get("empty_retrieval", 0),
        "low_streak": low_streak(),
        "threshold": eval_config.online_relevance_min,
        "created_at": latest.get("created_at"),
    }


def collect_signals(since: Optional[float] = None, with_online: bool = True) -> Dict[str, Any]:
    """采一轮全部信号（各来源互不影响：一个挂了其余照常）"""
    signals = {
        "quality": collect_quality(),
        "latency": collect_latency(since),
        "registry": _registry_brief(),
    }
    if with_online:
        signals["online"] = collect_online()
    return signals


def _registry_brief() -> Dict[str, Any]:
    """注册表摘要：有没有在建版本，直接决定这一轮能不能触发重建"""
    try:
        status = registry.registry_status() or {}
    except Exception as e:
        logger.error(f"读取索引注册表失败: {e}")
        return {"available": False, "error": str(e)[:200]}
    return {
        "available": bool(status.get("registry_available", True)),
        "logical": status.get("logical"),
        "alias": status.get("alias"),
        "active_version": (status.get("active") or {}).get("version"),
        "building_version": (status.get("building") or {}).get("version"),
        "write_targets": status.get("write_targets") or [],
    }


# ==================== 判定 ====================

def decide(signals: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """
    把信号翻译成结论
    :return: (rebuild_reasons, alert_reasons)
             rebuild_reasons 非空 = 建议重建；alert_reasons 只告警，重建解决不了
    """
    rebuild: List[str] = []
    alerts: List[str] = []

    quality = signals.get("quality") or {}
    if not quality.get("available"):
        alerts.append(f"索引质量指标采集失败: {quality.get('error') or '未知原因'}")
    else:
        sealed = quality.get("sealed_segments")
        if sealed and sealed > index_config.max_sealed_segments:
            rebuild.append(f"碎片化：sealed segment {sealed} > 阈值 {index_config.max_sealed_segments}")

        expected = index_config.dense_index.index_type
        actual = quality.get("dense_index_type")
        if actual and actual != expected:
            rebuild.append(f"索引类型 {actual} 与配置 {expected} 不符，这批数据没吃到调优参数")

        rows, registry_rows = quality.get("rows"), quality.get("registry_rows")
        if rows is not None and rows >= 0 and registry_rows:
            drift = abs(rows - registry_rows) / max(registry_rows, 1)
            if drift > 0.01:
                # 不放进 rebuild：重建只会把不一致的数据原样搬一遍，根因在导入/双写
                alerts.append(f"行数漂移 {drift:.1%}（实际 {rows} / 注册表 {registry_rows}），"
                              f"查导入日志与双写失败记录")

    latency = signals.get("latency") or {}
    baseline = monitor_config.p99_baseline_ms
    if latency.get("available") and baseline > 0:
        limit = baseline * index_config.p99_degrade_ratio
        p99 = latency.get("p99_latency_ms") or 0.0
        if p99 > limit:
            rebuild.append(f"p99 劣化：{p99:.0f}ms > 基线 {baseline:.0f}ms × "
                           f"{index_config.p99_degrade_ratio}（={limit:.0f}ms）")

    online = signals.get("online") or {}
    # stale = 本轮没评出新结果，看到的是库里的历史值。历史值只用于展示，不能拿来触发重建：
    # 线上评测一旦持续失败（裁判模型挂了、日志断流），几轮前的低相关率会一直留在库里，
    # 于是每 6 小时触发一次重建，却治不了真正的病
    if online.get("available") and not online.get("stale"):
        streak = online.get("low_streak") or 0
        if streak >= eval_config.online_low_streak:
            rebuild.append(f"线上相关率连续 {streak} 轮低于 {online.get('threshold')}"
                           f"（最近一轮 recall@3={online.get('recall@3')}）")
        # 空检索多说明知识库里没有用户要的东西，属于内容缺口
        empty = online.get("empty_retrieval") or 0
        sampled = online.get("sampled") or 0
        if empty and sampled and empty >= max(3, int(sampled * 0.2)):
            alerts.append(f"最近一轮有 {empty}/{sampled} 条查询完全没检索到内容，可能是知识库覆盖缺口")

    reg = signals.get("registry") or {}
    if not reg.get("available", True):
        # 注册表读不到就不知道有没有在建版本，此时触发重建可能和另一个重建互相双写
        alerts.append(f"索引注册表读不到（{reg.get('error') or 'Mongo 不可用'}），"
                      f"本轮无法确认是否有在建版本，判定不完整")
    elif reg.get("building_version"):
        alerts.append(f"已有在建版本 v{reg['building_version']}，本轮不会触发新的重建")

    return rebuild, alerts


# ==================== 执行 ====================

def rebuild_argv() -> List[str]:
    """自动触发重建时用的参数（与人工 CLI 完全一致，出问题可以照着复现）"""
    argv = ["--yes", "--mode", index_config.default_rebuild_mode,
            "--eval-limit", str(index_config.scheduler_eval_limit), "--drop-on-failure"]
    if index_config.scheduler_prepare_only:
        argv.append("--prepare-only")
    return argv


def trigger_rebuild(reasons: List[str]) -> Dict[str, Any]:
    """
    触发一次带门禁的重建
    :return: {"exit_code": int, "argv": [...], "message": str}
    """
    from processor.index_manager import rebuild

    argv = rebuild_argv()
    logger.warning(f"巡检判定需要重建，自动触发: {argv}\n原因:\n  - " + "\n  - ".join(reasons))
    started = time.time()
    try:
        code = rebuild.main(argv)
    except Exception as e:
        logger.error(f"自动重建异常: {e}", exc_info=True)
        return {"exit_code": 1, "argv": argv, "message": f"异常: {e}",
                "elapsed_seconds": round(time.time() - started, 1)}
    message = ("重建完成（停在 ready，等人工切别名）" if code == 0 and index_config.scheduler_prepare_only
               else "重建完成" if code == 0 else f"重建失败（退出码 {code}），详情见日志")
    logger.warning(f"自动重建结束: {message}")
    return {"exit_code": code, "argv": argv, "message": message,
            "elapsed_seconds": round(time.time() - started, 1)}


def run_cycle(sample: int = None, with_online: bool = True, force: bool = False,
              auto_rebuild: bool = None, lookback_hours: float = None) -> Dict[str, Any]:
    """
    跑一轮巡检
    :param sample: 本轮线上抽样条数（0 或 with_online=False 表示跳过线上评测）
    :param force: True 时即使已有在建版本也继续判定（默认跳过，避免两个重建互相双写）
    :param auto_rebuild: 覆盖 index_config.scheduler_auto_rebuild
    :return: 本轮记录（同时落 Mongo）
    """
    sample = index_config.scheduler_online_sample if sample is None else sample
    auto_rebuild = index_config.scheduler_auto_rebuild if auto_rebuild is None else auto_rebuild
    lookback_hours = lookback_hours or _LOOKBACK_HOURS_DEFAULT
    # 统计窗口从上一轮结束的地方接着算：每轮都从头扫 24 小时会让同一条查询被重复计入，
    # 慢查询率之类的指标就成了「最近 24 小时滑动平均」，看不出这一轮到底有没有变好
    since = last_cycle_ts() or (datetime.now() - timedelta(hours=lookback_hours)).timestamp()

    cycle: Dict[str, Any] = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "lookback_hours": lookback_hours,
        "auto_rebuild": bool(auto_rebuild),
    }

    # 1、先跑线上抽样评测：判定要用它的结果，所以必须在采信号之前
    online_ok = False
    if with_online and sample > 0:
        cycle["online_eval"] = _run_online_eval(sample, since)
        online_ok = not cycle["online_eval"].get("error")
    else:
        cycle["online_eval"] = {"skipped": True}

    # 2、采集其余信号
    cycle["signals"] = collect_signals(since=since, with_online=with_online and sample > 0)
    if not online_ok and cycle["signals"].get("online"):
        # 本轮没评出新结果，库里那份是历史值：留着展示，但 decide 不会拿它触发重建
        cycle["signals"]["online"]["stale"] = True

    # 3、判定
    rebuild_reasons, alert_reasons = decide(cycle["signals"])
    cycle["rebuild_reasons"] = rebuild_reasons
    cycle["alert_reasons"] = alert_reasons

    building = (cycle["signals"].get("registry") or {}).get("building_version")
    if rebuild_reasons and building and not force:
        # 已有在建版本：不再触发第二个（两个重建同时双写会互相干扰），只留结论
        cycle["decision"] = "skipped_building_in_progress"
        cycle["action"] = {"message": f"已有在建版本 v{building}，跳过触发；"
                                      f"如确认要重建请先处理它（--mark-failed 或等它完成）"}
    elif rebuild_reasons and auto_rebuild:
        cycle["decision"] = "rebuild_triggered"
        cycle["action"] = trigger_rebuild(rebuild_reasons)
    elif rebuild_reasons:
        cycle["decision"] = "rebuild_recommended"
        argv = ["python", "-m", "processor.index_manager.rebuild"] + rebuild_argv()
        cycle["action"] = {"message": "建议重建（未自动执行）", "command": " ".join(argv)}
    elif alert_reasons:
        cycle["decision"] = "alert_only"
        cycle["action"] = {"message": "有告警但不需要重建"}
    else:
        cycle["decision"] = "healthy"
        cycle["action"] = {"message": "索引健康，无需处理"}

    cycle["ts"] = time.time()
    cycle["created_at"] = datetime.now().isoformat(timespec="seconds")
    save_cycle(cycle)
    return cycle


def _run_online_eval(sample: int, since: float) -> Dict[str, Any]:
    """
    跑一轮线上抽样评测，返回精简结果
    这里直接调函数而不是起子进程：同一进程内可以复用已加载的模型与连接，
    也省掉一次「解析子进程输出」的脆弱环节
    """
    from eval import online_eval

    since_hours = max((time.time() - since) / 3600.0, 1.0)
    try:
        result = online_eval.run(sample=sample, since_hours=since_hours)
    except Exception as e:
        logger.error(f"线上抽样评测失败: {e}", exc_info=True)
        return {"error": str(e)[:200]}

    if result.get("error"):
        return {"error": result["error"]}

    online_eval.write_result_file(result)
    online_eval.save_run(result)
    return {
        "run_id": result.get("run_id"),
        "sampled": result.get("sampled"),
        "judged": result.get("judged"),
        "judge_rate": result.get("judge_rate"),
        "recall@1": result.get("recall@1"),
        "recall@3": result.get("recall@3"),
        "useful_rate": result.get("useful_rate"),
        "by_status": result.get("by_status"),
        "low_streak": online_eval.low_streak(),
    }


# ==================== 持久化 ====================

def _coll():
    return get_collection(COLLECTION_NAME, INDEXES)


def save_cycle(cycle: Dict[str, Any]) -> Optional[str]:
    """落一轮巡检记录（跨进程可见，/admin/scheduler/* 与 --status 都读它）"""
    coll = _coll()
    if coll is None:
        logger.warning("运维 Mongo 不可用，巡检记录不落库")
        return None
    try:
        return str(coll.insert_one(dict(cycle)).inserted_id)
    except Exception as e:
        logger.error(f"巡检记录落库失败: {e}")
        return None


def recent_cycles(limit: int = 5) -> List[Dict[str, Any]]:
    """最近几轮巡检记录"""
    coll = _coll()
    if coll is None:
        return []
    try:
        return [as_dict(d) for d in coll.find().sort("ts", DESCENDING).limit(limit)]
    except Exception as e:
        logger.error(f"读取巡检历史失败: {e}")
        return []


def last_cycle_ts() -> Optional[float]:
    """上一轮的时间戳（决定本轮 query_log 统计从哪儿开始）"""
    cycles = recent_cycles(1)
    return cycles[0].get("ts") if cycles else None


# ==================== 常驻循环 ====================

def loop(interval_hours: float = None, **kwargs) -> None:
    """
    常驻巡检：每 interval_hours 跑一轮
    每轮各自兜异常——一轮失败不该让整个巡检线程死掉（那等于监控悄悄停摆）
    """
    interval_hours = interval_hours or index_config.scheduler_interval_hours
    interval = max(interval_hours * 3600, 60)
    logger.info(f"索引巡检已启动，每 {interval_hours} 小时一轮（Ctrl+C 退出）")
    while True:
        try:
            cycle = run_cycle(**kwargs)
            logger.warning(f"巡检结论: {cycle['decision']} | {cycle['action'].get('message')}")
        except Exception as e:
            logger.error(f"巡检轮次异常，{interval_hours} 小时后重试: {e}", exc_info=True)
        next_at = datetime.now() + timedelta(seconds=interval)
        logger.info(f"下一轮巡检: {next_at.isoformat(timespec='seconds')}")
        time.sleep(interval)


# ==================== 输出 ====================

def format_report(cycle: Dict[str, Any]) -> str:
    """排版成给人看的文本"""
    signals = cycle.get("signals") or {}
    quality = signals.get("quality") or {}
    latency = signals.get("latency") or {}
    online = signals.get("online") or {}
    reg = signals.get("registry") or {}
    lines = [
        f"索引巡检 {cycle.get('run_id')} | 结论: {cycle.get('decision')}",
        f"  索引: {quality.get('collection')} v{quality.get('version')} | "
        f"行数 {quality.get('rows')} | sealed {quality.get('sealed_segments')} | "
        f"类型 {quality.get('dense_index_type')}",
        f"  延迟: 查询 {latency.get('total')} 条 | p99 {latency.get('p99_latency_ms')}ms | "
        f"慢查询率 {latency.get('slow_rate')}",
        f"  线上: recall@3 {online.get('recall@3')}（阈值 {online.get('threshold')}）| "
        f"连续低于阈值 {online.get('low_streak') or 0} 轮 | 判定率 {online.get('judge_rate')}",
        f"  注册表: alias={reg.get('alias')} active=v{reg.get('active_version')} "
        f"building=v{reg.get('building_version')}",
    ]
    online_eval_info = cycle.get("online_eval") or {}
    if online_eval_info.get("error"):
        lines.append(f"  [!] 本轮线上抽样评测未完成: {online_eval_info['error']}")
    elif online_eval_info.get("skipped"):
        lines.append("  本轮跳过线上抽样评测")
    if online.get("stale"):
        lines.append("  [!] 线上信号是历史值（本轮没评出新结果），不参与重建判定")
    if cycle.get("rebuild_reasons"):
        lines.append("  [!] 建议重建:")
        lines.extend(f"    - {r}" for r in cycle["rebuild_reasons"])
    if cycle.get("alert_reasons"):
        lines.append("  告警（重建解决不了）:")
        lines.extend(f"    - {a}" for a in cycle["alert_reasons"])
    action = cycle.get("action") or {}
    lines.append(f"  动作: {action.get('message')}")
    if action.get("command"):
        lines.append(f"  一键执行: {action['command']}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="索引巡检调度器：采集信号 → 判定 → 可选自动重建")
    parser.add_argument("--loop", action="store_true",
                        help=f"常驻运行，每 --interval-hours 一轮（默认 {index_config.scheduler_interval_hours}）")
    parser.add_argument("--interval-hours", type=float, help="常驻模式的巡检周期")
    parser.add_argument("--sample", type=int, help=f"每轮线上抽样条数，0 表示跳过（默认 "
                                                  f"{index_config.scheduler_online_sample}）")
    parser.add_argument("--no-online", action="store_true", help="跳过线上抽样评测，只看索引侧信号")
    parser.add_argument("--lookback-hours", type=float, help="query_log 统计窗口，默认 24 小时")
    parser.add_argument("--force", action="store_true", help="已有在建版本时仍然触发重建")
    parser.add_argument("--auto-rebuild", dest="auto_rebuild", action="store_true",
                        help="判定需要重建时自动执行（默认只给结论和命令）")
    parser.add_argument("--no-auto-rebuild", dest="auto_rebuild", action="store_false",
                        help="强制只给结论，即使 .env 里开了自动重建")
    parser.set_defaults(auto_rebuild=None)
    parser.add_argument("--status", type=int, nargs="?", const=5, metavar="N",
                        help="只打印最近 N 轮巡检记录（默认 5）后退出")
    parser.add_argument("--json", action="store_true", help="输出 JSON 而不是排版文本")
    args = parser.parse_args(argv)

    if args.status is not None:
        cycles = recent_cycles(args.status)
        if not cycles:
            print("还没有巡检记录（Mongo 不可用或从未跑过 --run）")
            return 1
        if args.json:
            print(json.dumps(cycles, ensure_ascii=False, indent=2, default=str))
        else:
            for cycle in cycles:
                print(format_report(cycle))
                print("-" * 60)
        return 0

    if args.loop:
        loop(interval_hours=args.interval_hours, sample=args.sample,
             with_online=not args.no_online, force=args.force,
             auto_rebuild=args.auto_rebuild, lookback_hours=args.lookback_hours)
        return 0

    cycle = run_cycle(sample=args.sample, with_online=not args.no_online, force=args.force,
                      auto_rebuild=args.auto_rebuild, lookback_hours=args.lookback_hours)
    if args.json:
        print(json.dumps(cycle, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_report(cycle))
    # 退出码：0=健康，2=有告警但不需重建，3=建议/已触发重建，1=执行失败
    if cycle["decision"] in ("rebuild_recommended", "rebuild_triggered"):
        return 3 if (cycle.get("action") or {}).get("exit_code", 0) == 0 else 1
    if cycle["decision"] == "alert_only":
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
