# eval/online_eval.py

"""线上抽样评测：拿真实查询当评测集，用 LLM 当裁判打相关性分。

为什么离线评测还不够
--------------------
离线评测集是「从语料反向生成的问题」，它衡量的是检索系统对**理想问题**的表现。
真实用户的问题往往更短、更口语、带错别字、甚至问的是知识库里根本没有的东西。
两者之间的差距只能靠线上数据补：从 query_log 里抽样真实查询，看当时真正返回给
大模型的那几个切片到底相不相关。

打分的对象是「检索结果」而不是「最终答案」：答案质量还受 prompt、模型、上下文截断
影响，混在一起就分不清是检索的锅还是生成的锅。这一层只回答一个问题——
**该找的资料有没有被找出来**。

评分口径（0/1/2）
    2 = 直接能回答用户的问题
    1 = 相关但不足以回答（沾边、只覆盖一部分）
    0 = 无关

产出去向
    · Mongo `eval_online`：每轮一条汇总 + 逐条明细，调度器靠它判断「连续几轮低于阈值」
    · eval/results/{ts}_online.json：与离线评测同构，指标采集器据此产出
      `kb_eval_recall{scenario="online",mode="online"}`，看板与告警直接复用

注意：这里的 recall@k **不是**离线评测那个 recall@k（没有 ground truth 就算不出真召回率），
而是「前 k 条里有没有相关资料」的比例，是个代理指标。看板上两者靠 mode 标签区分。
"""

import argparse
import json
import sys
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pymongo import DESCENDING

from config.milvus_config import milvus_config
from eval.config import eval_config
from tool.logger import logger
from utils.llm_utils import get_llm_client
from utils.milvus_utils import escape_milvus_string, get_milvus_client
from utils.mongo_ops_utils import as_dict, get_collection
from utils.query_log_utils import sample_queries

COLLECTION_NAME = "eval_online"
INDEXES = [
    ([("ts", DESCENDING)], {}),
]

# 裁判提示词。要求逐条打分 + 给一句理由：
# 只要分数不要理由时，模型很容易一律给高分（没有可核对的依据），理由能显著降低这种倾向
JUDGE_PROMPT = """你是一个检索质量评审员。下面是用户在产品知识库里提出的真实问题，以及检索系统返回给他的若干条资料片段。
请逐条判断每条资料对回答这个问题的帮助程度。

打分标准：
2 = 这条资料能直接回答用户的问题（含所需参数/步骤/结论）
1 = 相关但不足以回答（只沾边、只覆盖问题的一部分、或讲的是同系列的其它型号）
0 = 与问题无关

用户问题：
{query}

资料片段：
{passages}

只输出 JSON，格式如下（scores 的长度必须与资料片段条数一致，顺序对应）：
{{"scores": [2, 0, 1], "reason": "一句话说明判断依据"}}
"""


# ==================== 取回当时真正返回的资料 ====================

def fetch_passages(triples: List[tuple], collection: str, client=None) -> Dict[tuple, str]:
    """
    按业务三元组 (file_title, title, part) 回库取正文
    :return: {三元组: 正文}；取不到的三元组不出现在结果里

    为什么要回库：query_log 刻意不存正文（日志集合会迅速膨胀到没法查），
    只存了三元组，评测时按三元组回查一次。三元组在业务上是唯一定位一段资料的键，
    所以即使索引重建过、chunk_id 全变了，历史日志依然可用——这正是当初选它的原因。
    """
    client = client if client is not None else get_milvus_client()
    if not client or not triples:
        return {}

    clauses = []
    for file_title, title, part in triples:
        ft = escape_milvus_string(str(file_title or ""))
        tt = escape_milvus_string(str(title or ""))
        clauses.append(f'(file_title == "{ft}" && title == "{tt}" && part == {int(part or 0)})')

    try:
        rows = client.query(
            collection_name=collection,
            filter=" || ".join(clauses),
            output_fields=["file_title", "title", "part", "content"],
            limit=len(clauses),
        )
    except Exception as e:
        logger.error(f"回查切片正文失败（{len(clauses)} 条）: {e}")
        return {}

    found = {}
    for row in rows or []:
        key = (row.get("file_title"), row.get("title"), int(row.get("part") or 0))
        found[key] = (row.get("content") or "")[:eval_config.online_content_chars]
    return found


def triples_of(log: dict, top: int) -> List[tuple]:
    """从一条 query_log 里取出当时交给大模型的前 top 个切片的三元组"""
    out = []
    for doc in (log.get("reranked") or [])[:top]:
        out.append((doc.get("file_title"), doc.get("title"), int(doc.get("part") or 0)))
    return out


# ==================== LLM 裁判 ====================

def judge(query: str, passages: List[str]) -> Optional[Dict[str, Any]]:
    """
    让 LLM 给一组资料打分
    :return: {"scores": [...], "reason": "..."}；调用或解析失败返回 None（算未评）
    """
    blocks = "\n\n".join(f"【资料 {i + 1}】\n{text}" for i, text in enumerate(passages))
    prompt = JUDGE_PROMPT.format(query=query, passages=blocks)

    try:
        llm = get_llm_client(model=eval_config.online_judge_model or None, json_mode=True)
        content = llm.invoke(prompt).content
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content)
        scores = [int(s) for s in (parsed.get("scores") or [])]
        # 条数对不上说明模型没按格式答，这种结果不能用（宁可算未评也不要脏数据）
        if len(scores) != len(passages):
            logger.warning(f"裁判返回条数不符（{len(scores)}/{len(passages)}），丢弃该条")
            return None
        scores = [min(max(s, 0), 2) for s in scores]
        return {"scores": scores, "reason": str(parsed.get("reason") or "")[:300]}
    except Exception as e:
        logger.error(f"LLM 裁判失败 query={query!r}: {e}")
        return None


# ==================== 一轮抽样评测 ====================

def run(sample: int = None, since_hours: float = None, top: int = None,
        collection: str = None, seed: int = None, dry_run: bool = False) -> Dict[str, Any]:
    """
    跑一轮线上抽样评测
    :param sample: 抽样条数，默认取配置
    :param since_hours: 只从最近 N 小时的日志里抽（默认 24h；日志少时可放宽）
    :param top: 每条查询评前几条切片，默认取配置
    :param collection: 回查正文的集合，默认 .env 的 CHUNKS_COLLECTION（= 别名）
    :param seed: 抽样随机种子（复现用）
    :param dry_run: 只抽样打印，不调 LLM、不落库
    :return: 汇总字典
    """
    sample = sample or eval_config.online_sample
    top = top or eval_config.online_judge_top
    collection = collection or milvus_config.chunks_collection
    since_hours = eval_config.online_since_hours if since_hours is None else since_hours
    since = (datetime.now() - timedelta(hours=since_hours)).timestamp() if since_hours else None

    logs = sample_queries(limit=sample, since=since,
                          exclude_intents=eval_config.online_exclude_intents, seed=seed)
    started = time.time()
    result: Dict[str, Any] = {
        "run_id": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "sampled": len(logs),
        "since_hours": since_hours,
        "top": top,
        "collection": collection,
        "dry_run": dry_run,
        "rows": [],
    }
    if not logs:
        result["error"] = f"最近 {since_hours or '不限'} 小时内没有可抽样的查询日志"
        logger.warning(result["error"])
        return result

    client = None if dry_run else get_milvus_client()
    if not dry_run and client is None:
        result["error"] = "Milvus 连接失败，无法回查切片正文"
        logger.error(result["error"])
        return result

    for idx, log in enumerate(logs, start=1):
        query = (log.get("rewritten_query") or log.get("original_query") or "").strip()
        triples = triples_of(log, top)
        row: Dict[str, Any] = {
            "query": query[:200],
            "intent": log.get("intent") or "UNKNOWN",
            "triples": [list(t) for t in triples],
            "total_latency_ms": log.get("total_latency_ms"),
            "slow": bool(log.get("slow")),
        }

        if not query:
            row["status"] = "no_query"
            result["rows"].append(row)
            continue
        if not triples:
            # 一条都没检索到：这本身就是最有价值的线上信号（知识库里可能真没有）
            row["status"] = "empty_retrieval"
            result["rows"].append(row)
            logger.info(f"[{idx}/{len(logs)}] 空检索: {query[:40]}")
            continue
        if dry_run:
            row["status"] = "dry_run"
            result["rows"].append(row)
            print(f"[{idx}/{len(logs)}] {row['intent']:<10} {query[:50]} → {len(triples)} 条待评")
            continue

        passages_map = fetch_passages(triples, collection, client=client)
        passages = [passages_map[t] for t in triples if passages_map.get(t)]
        if not passages:
            row["status"] = "content_missing"
            result["rows"].append(row)
            logger.warning(f"[{idx}/{len(logs)}] 正文回查不到（索引重建过或已删除）: {query[:40]}")
            continue

        verdict = judge(query, passages)
        if not verdict:
            row["status"] = "unjudged"
            result["rows"].append(row)
            continue

        row.update(status="judged", scores=verdict["scores"], reason=verdict["reason"],
                   judged=len(passages))
        result["rows"].append(row)
        if idx % 10 == 0 or idx == len(logs):
            logger.info(f"进度 {idx}/{len(logs)} | 已评 {sum(1 for r in result['rows'] if r['status'] == 'judged')}")

    summarize(result)
    result["elapsed_seconds"] = round(time.time() - started, 2)
    return result


def summarize(result: Dict[str, Any]) -> Dict[str, Any]:
    """
    把逐条明细汇总成指标（写回 result 并返回）
    口径：
      recall@1 = top1 相关（>=1 分）的查询占比   —— 「第一条就给了有用的东西」
      recall@3 = 前 3 条里至少一条相关的查询占比 —— 「这轮检索到底有没有产出」
      useful_rate = 所有被打分的资料里 2 分的占比 —— 「找出来的东西有多硬」
    """
    rows = result.get("rows") or []
    judged = [r for r in rows if r.get("status") == "judged"]
    by_status: Dict[str, int] = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1

    top1_hit = top3_hit = 0
    score_hist = {0: 0, 1: 0, 2: 0}
    by_intent: Dict[str, Dict[str, int]] = {}
    for r in judged:
        scores = r.get("scores") or []
        for s in scores:
            score_hist[min(max(int(s), 0), 2)] += 1
        if scores and scores[0] >= 1:
            top1_hit += 1
        if any(s >= 1 for s in scores):
            top3_hit += 1
        bucket = by_intent.setdefault(r.get("intent") or "UNKNOWN", {"judged": 0, "top3_hit": 0})
        bucket["judged"] += 1
        bucket["top3_hit"] += 1 if any(s >= 1 for s in scores) else 0

    n = len(judged)
    total_docs = sum(score_hist.values())
    summary = {
        "judged": n,
        "by_status": by_status,
        "recall@1": round(top1_hit / n, 4) if n else None,
        "recall@3": round(top3_hit / n, 4) if n else None,
        "useful_rate": round(score_hist[2] / total_docs, 4) if total_docs else None,
        "score_hist": {str(k): v for k, v in score_hist.items()},
        "judged_docs": total_docs,
        "by_intent": {k: {**v, "rate": round(v["top3_hit"] / v["judged"], 4) if v["judged"] else None}
                      for k, v in by_intent.items()},
        # 有效判定率太低说明这轮结果不可信（Milvus 回查失败 / LLM 格式错误太多）
        "judge_rate": round(n / len(rows), 4) if rows else 0.0,
    }
    result.update(summary)
    return summary


# ==================== 持久化 ====================

def _coll():
    return get_collection(COLLECTION_NAME, INDEXES)


def save_run(result: Dict[str, Any]) -> Optional[str]:
    """写一条评测记录到 Mongo（失败只记日志，评测结果仍在返回值和文件里）"""
    coll = _coll()
    if coll is None:
        logger.warning("运维 Mongo 不可用，本轮线上评测结果不落库（仍写结果文件）")
        return None
    doc = dict(result)
    doc["ts"] = time.time()
    doc["created_at"] = datetime.now().isoformat(timespec="seconds")
    try:
        return str(coll.insert_one(doc).inserted_id)
    except Exception as e:
        logger.error(f"线上评测结果落库失败: {e}")
        return None


def recent_runs(limit: int = 10) -> List[Dict[str, Any]]:
    """最近几轮线上评测（调度器判断「连续低于阈值」要用）"""
    coll = _coll()
    if coll is None:
        return []
    try:
        return [as_dict(d) for d in coll.find().sort("ts", DESCENDING).limit(limit)]
    except Exception as e:
        logger.error(f"读取线上评测历史失败: {e}")
        return []


def low_streak(threshold: float = None, limit: int = 10) -> int:
    """
    最近连续多少轮「相关率低于阈值」
    为什么要连续：单轮抽样只有几十条，抖动很大；连续两轮都低才说明是真劣化而不是运气差
    :return: 连续轮数（0 表示最近一轮是达标的）
    """
    threshold = eval_config.online_relevance_min if threshold is None else threshold
    streak = 0
    for run_doc in recent_runs(limit):
        rate = run_doc.get("recall@3")
        # 未评出结果的轮次不算达标也不算不达标：直接跳过，避免 LLM 抽风触发重建
        if rate is None:
            continue
        if rate < threshold:
            streak += 1
        else:
            break
    return streak


def write_result_file(result: Dict[str, Any]) -> str:
    """
    写成与离线评测同构的结果文件，让指标采集器能直接读到
    recall@k 键名沿用离线口径（含义见文件头说明），mode 标签区分两者
    """
    eval_config.result_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = eval_config.result_dir / f"{ts}_online.json"

    by_scenario = {}
    metrics = {k: result.get(k) for k in ("recall@1", "recall@3") if result.get(k) is not None}
    if metrics:
        by_scenario["online"] = metrics
        for intent, bucket in (result.get("by_intent") or {}).items():
            if bucket.get("rate") is not None:
                by_scenario[f"online_{intent}"] = {"recall@3": bucket["rate"]}

    payload = {
        "meta": {
            "tag": "online",
            "mode": "online",
            "created_at": result.get("run_id"),
            "sampled": result.get("sampled"),
            "judged": result.get("judged"),
            "judge_rate": result.get("judge_rate"),
            "useful_rate": result.get("useful_rate"),
            "by_status": result.get("by_status"),
            "collection": result.get("collection"),
            "note": "recall@k 为线上相关率代理指标，非离线真召回率",
        },
        "by_scenario": by_scenario,
        "overall": metrics,
        "details": result.get("rows") or [],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(path)


def format_report(result: Dict[str, Any]) -> str:
    """排版成给人看的文本"""
    if result.get("error"):
        return f"线上抽样评测未完成：{result['error']}"
    lines = [
        f"线上抽样评测 {result.get('run_id')} | 抽样 {result.get('sampled')} 条 | "
        f"有效判定 {result.get('judged')} 条（判定率 {result.get('judge_rate'):.0%}）",
        f"  top1 相关率: {result.get('recall@1')}",
        f"  top3 相关率: {result.get('recall@3')}（阈值 {eval_config.online_relevance_min}）",
        f"  强相关占比: {result.get('useful_rate')} | 分数分布: {result.get('score_hist')}",
        f"  状态分布: {result.get('by_status')}",
    ]
    for intent, bucket in (result.get("by_intent") or {}).items():
        lines.append(f"  {intent}: {bucket.get('top3_hit')}/{bucket.get('judged')} = {bucket.get('rate')}")
    streak = low_streak()
    if streak:
        lines.append(f"  [!] 已连续 {streak} 轮低于阈值（连续 {eval_config.online_low_streak} 轮会触发重建判定）")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="线上抽样评测：用 LLM 给真实查询的检索结果打相关性分")
    parser.add_argument("--sample", type=int, help=f"抽样条数，默认 {eval_config.online_sample}")
    parser.add_argument("--since-hours", type=float,
                        help=f"只抽最近 N 小时的日志，默认 {eval_config.online_since_hours}")
    parser.add_argument("--top", type=int, help=f"每条查询评前几条切片，默认 {eval_config.online_judge_top}")
    parser.add_argument("--collection", help="回查正文的集合，默认 .env 的 CHUNKS_COLLECTION")
    parser.add_argument("--seed", type=int, help="抽样随机种子（复现用）")
    parser.add_argument("--dry-run", action="store_true", help="只抽样打印，不调 LLM 也不落库")
    parser.add_argument("--no-save", action="store_true", help="不落 Mongo（仍写结果文件）")
    parser.add_argument("--json", action="store_true", help="输出 JSON 而不是排版文本")
    args = parser.parse_args(argv)

    result = run(sample=args.sample, since_hours=args.since_hours, top=args.top,
                 collection=args.collection, seed=args.seed, dry_run=args.dry_run)

    if result.get("error"):
        print(result["error"])
        return 2
    if not result.get("rows"):
        print("没有可评的查询")
        return 2

    if not args.dry_run:
        path = write_result_file(result)
        result["result_file"] = path
        logger.info(f"结果文件: {path}")
        if not args.no_save:
            result["mongo_id"] = save_run(result)
        # 判定率过低说明这轮数据不可信，退出码区分开，调度器据此决定要不要采信
        if (result.get("judge_rate") or 0) < 0.5:
            logger.error(f"有效判定率仅 {result.get('judge_rate'):.0%}，本轮结果不采信")
            if not args.json:
                print(format_report(result))
            return 3

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_report(result))
    return 0


if __name__ == "__main__":
    sys.exit(main())
