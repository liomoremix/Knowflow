# processor/index_manager/validate.py

"""新旧索引一致性校验——蓝绿切换前的最后一道闸门。

校验什么
--------
1. 行数一致：搬运模式下新旧集合行数必须相等，少了就是漏搬，多了就是重复搬。
2. 召回一致：同一批评测查询、**同一个查询向量**打到两个集合上，gold 命中率差不能超过 1 个百分点。
3. 结果一致：两边 top10 的业务三元组集合 Jaccard 相似度达标（搬运模式要求几乎完全一样）。
4. 双写可见：重建期间导入的文件（file_title）在新集合里必须查得到。

为什么关掉 HyDE
--------------
HyDE 会让大模型现编一段假设性文档再去检索，两边跑两次得到的假设文档不一样，
差异就分不清是「索引不一致」还是「HyDE 随机性」。校验只关心索引，所以关掉。
同理，两边必须复用同一个查询向量，否则 embedding 的浮点抖动也会污染结论。

任一项不过 → 不切换，把在建版本标成 failed 留给人工。
"""

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

from config.index_config import index_config
from eval.config import all_dataset_paths
from eval.dataset import gold_key, load_records
from eval.metrics import rank_of_gold
from eval.retrieve import resolve_strategy, retrieve_raw, to_ranked_docs
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.milvus_utils import count_rows, escape_milvus_string, get_milvus_client

# 一致性校验固定用的 top-k（Jaccard 比较的是这个范围内的三元组集合）
VALIDATE_TOP_K = 10
# 双写可见性检查最多查多少个 file_title（文件多时抽样，避免校验跑太久）
MAX_TITLES_CHECKED = 200

CHECK_ROW_COUNT = "row_count"
CHECK_HIT_RATE = "gold_hit_rate"
CHECK_JACCARD = "top10_jaccard"
CHECK_TITLES = "double_write_visibility"


# ==================== 取样 ====================

def samples_from_dataset(size: int, seed: int = 42) -> list[dict]:
    """从评测集取样（首选：查询是真实用户问法，gold 是标注好的）"""
    records = []
    for path in all_dataset_paths():
        records.extend(load_records(path))
    if not records:
        return []

    rng = random.Random(seed)
    rng.shuffle(records)
    return [
        {
            "qid": record.get("qid"),
            "scenario": record.get("scenario"),
            "query": record.get("query"),
            "gold": gold_key(record),
            "item_name": record.get("gold_item_name"),
        }
        for record in records[:size]
    ]


def samples_from_collection(client, collection: str, size: int, seed: int = 42) -> list[dict]:
    """
    没有评测集时的兜底：直接抽切片，把正文当查询、把自己的三元组当 gold。
    这是弱测试（查询≈原文，几乎必命中），只能证明索引没坏，不能证明召回质量，
    所以结果里会标明来源，别拿它当评测报告用。
    """
    rows = client.query(
        collection_name=collection,
        filter="",
        output_fields=["content", "title", "parent_title", "file_title", "part", "item_name"],
        limit=min(size * 3, 1000),
    )
    rows = [r for r in rows if (r.get("content") or "").strip()]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return [
        {
            "qid": f"probe-{idx}",
            "scenario": "content_probe",
            "query": row["content"][:500],
            "gold": (row.get("file_title"), row.get("title"), int(row.get("part") or 0)),
            "item_name": row.get("item_name"),
        }
        for idx, row in enumerate(rows[:size])
    ]


def collect_samples(client, old_collection: str, size: int, seed: int = 42,
                    require_dataset: bool = False) -> tuple[list[dict], str]:
    """取样并返回 (样本列表, 来源标记)"""
    samples = samples_from_dataset(size, seed)
    if samples:
        return samples, "dataset"
    if require_dataset:
        return [], "dataset_missing"
    logger.warning("评测集为空，退化成用切片正文当查询（弱校验，只证明索引没坏）")
    return samples_from_collection(client, old_collection, size, seed), "content_probe"


# ==================== 各项校验 ====================

def _jaccard(left: set, right: set) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _triples(docs: list[dict], top_k: int) -> set:
    return {
        (d.get("file_title"), d.get("title"), int(d.get("part") or 0))
        for d in docs[:top_k]
    }


def check_retrieval(samples: list[dict], old: str, new: str, ef_search: int = None) -> dict:
    """
    对同一批样本在两个集合上跑召回层，比较命中率与 top10 结果集
    :return: {old_hit_rate, new_hit_rate, hit_rate_diff, jaccard_mean, jaccard_min, worst[]}
    """
    strategy = resolve_strategy(ef_search=ef_search)
    # 只比索引：HyDE 引入大模型随机性，必须关掉
    strategy["enable_hyde"] = False

    old_hits = new_hits = 0
    jaccards = []
    worst = []

    for idx, sample in enumerate(samples, start=1):
        query = sample["query"]
        gold = sample["gold"]
        item_names = [sample["item_name"]] if sample.get("item_name") else None

        # 同一个查询向量打两边，排除 embedding 抖动的干扰
        embeddings = generate_embeddings([query])
        reuse = {"query_dense_vec": embeddings["dense"][0],
                 "query_sparse_vec": embeddings["sparse"][0]}

        old_docs = to_ranked_docs(retrieve_raw(query, strategy, item_names, collection=old, reuse=reuse))
        new_docs = to_ranked_docs(retrieve_raw(query, strategy, item_names, collection=new, reuse=reuse))

        old_rank = rank_of_gold(gold, old_docs)
        new_rank = rank_of_gold(gold, new_docs)
        old_hits += int(0 < old_rank <= VALIDATE_TOP_K)
        new_hits += int(0 < new_rank <= VALIDATE_TOP_K)

        score = _jaccard(_triples(old_docs, VALIDATE_TOP_K), _triples(new_docs, VALIDATE_TOP_K))
        jaccards.append(score)
        if score < 1.0:
            worst.append({"qid": sample.get("qid"), "query": query, "jaccard": round(score, 4),
                          "old_rank": old_rank, "new_rank": new_rank})

        if idx % 10 == 0:
            logger.info(f"一致性校验进度 {idx}/{len(samples)}")

    total = len(samples) or 1
    worst.sort(key=lambda x: x["jaccard"])
    return {
        "sample_size": len(samples),
        "old_hit_rate": round(old_hits / total, 4),
        "new_hit_rate": round(new_hits / total, 4),
        "hit_rate_diff": round(abs(new_hits - old_hits) / total, 4),
        "jaccard_mean": round(sum(jaccards) / total, 4) if jaccards else 0.0,
        "jaccard_min": round(min(jaccards), 4) if jaccards else 0.0,
        "worst": worst[:10],
    }


def check_file_titles(client, old: str, new: str) -> dict:
    """
    双写可见性：旧集合里的每个 file_title，在新集合里都要查得到。
    查不到说明这批数据是在双写开启前导入、又没被 backfill 覆盖到——切过去就会丢文档。
    """
    rows = client.query(collection_name=old, filter="", output_fields=["file_title"], limit=MAX_TITLES_CHECKED * 5)
    titles = sorted({r.get("file_title") for r in rows if r.get("file_title")})[:MAX_TITLES_CHECKED]

    missing = []
    for title in titles:
        escaped = escape_milvus_string(title)
        found = client.query(collection_name=new, filter=f"file_title == '{escaped}'",
                             output_fields=["chunk_id"], limit=1)
        if not found:
            missing.append(title)

    return {"titles_checked": len(titles), "missing": missing, "missing_count": len(missing)}


# ==================== 汇总 ====================

def validate(old: str, new: str, mode: str = None, samples: int = None, seed: int = 42,
             require_dataset: bool = False, ef_search: int = None) -> dict:
    """
    跑完整的一致性校验
    :param old: 旧物理集合名（当前 active）
    :param new: 新物理集合名（在建）
    :param mode: move_vectors / re_embed，决定 Jaccard 门槛
    :param samples: 取样条数
    :return: 校验结果字典，含 passed 与每项 check 的明细
    """
    mode = mode or index_config.default_rebuild_mode
    samples = samples or index_config.validate_sample_size
    client = get_milvus_client()

    checks = []

    # --- 1. 行数 ---
    old_rows = count_rows(client, old)
    new_rows = count_rows(client, new)
    row_ok = old_rows >= 0 and old_rows == new_rows
    checks.append({
        "name": CHECK_ROW_COUNT,
        "passed": row_ok,
        "detail": f"旧={old_rows} 新={new_rows}",
        "expected": "两边行数相等",
    })

    # --- 2/3. 召回一致 + 结果一致 ---
    sample_list, source = collect_samples(client, old, samples, seed, require_dataset)
    if not sample_list:
        checks.append({"name": CHECK_HIT_RATE, "passed": False,
                       "detail": "没有可用样本（评测集为空且 --require-dataset）",
                       "expected": "至少 1 条样本"})
        retrieval = {}
    else:
        retrieval = check_retrieval(sample_list, old, new, ef_search)

        hit_ok = retrieval["hit_rate_diff"] <= index_config.validate_hit_rate_max_diff
        checks.append({
            "name": CHECK_HIT_RATE,
            "passed": hit_ok,
            "detail": f"旧={retrieval['old_hit_rate']} 新={retrieval['new_hit_rate']} "
                      f"差={retrieval['hit_rate_diff']}",
            "expected": f"差 ≤ {index_config.validate_hit_rate_max_diff}",
        })

        jaccard_floor = (index_config.validate_jaccard_min_move if mode == "move_vectors"
                         else index_config.validate_jaccard_min)
        jaccard_ok = retrieval["jaccard_mean"] >= jaccard_floor
        checks.append({
            "name": CHECK_JACCARD,
            "passed": jaccard_ok,
            "detail": f"均值={retrieval['jaccard_mean']} 最小={retrieval['jaccard_min']}",
            "expected": f"均值 ≥ {jaccard_floor}（mode={mode}）",
        })

    # --- 4. 双写可见性 ---
    titles = check_file_titles(client, old, new)
    checks.append({
        "name": CHECK_TITLES,
        "passed": titles["missing_count"] == 0,
        "detail": f"检查 {titles['titles_checked']} 个 file_title，缺失 {titles['missing_count']} 个"
                  + (f": {titles['missing'][:5]}" if titles["missing"] else ""),
        "expected": "旧集合里的文件在新集合都能查到",
    })

    result = {
        "passed": all(c["passed"] for c in checks),
        "mode": mode,
        "sample_source": source,
        "old": {"collection": old, "rows": old_rows},
        "new": {"collection": new, "rows": new_rows},
        "checks": checks,
        "retrieval": retrieval,
        "titles": titles,
        "ts": datetime.now().isoformat(timespec="seconds"),
    }
    logger.info(f"一致性校验{'通过' if result['passed'] else '未通过'}: {old} → {new}")
    return result


def format_report(result: dict) -> str:
    """把校验结果排版成给人看的文本"""
    lines = [
        f"一致性校验 {'[OK] 通过' if result['passed'] else '[FAIL] 未通过'}",
        f"  旧集合: {result['old']['collection']} ({result['old']['rows']} 行)",
        f"  新集合: {result['new']['collection']} ({result['new']['rows']} 行)",
        f"  模式: {result['mode']}   样本来源: {result['sample_source']}",
        "",
    ]
    for check in result["checks"]:
        lines.append(f"  [{'PASS' if check['passed'] else 'FAIL'}] {check['name']}")
        lines.append(f"         实际: {check['detail']}")
        lines.append(f"         要求: {check['expected']}")

    worst = (result.get("retrieval") or {}).get("worst") or []
    if worst:
        lines.append("")
        lines.append("  差异最大的样本（Jaccard < 1）:")
        for item in worst[:5]:
            lines.append(f"    {item['jaccard']:.4f}  old_rank={item['old_rank']} "
                         f"new_rank={item['new_rank']}  {item['query'][:40]!r}")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="新旧索引一致性校验（蓝绿切换前的闸门）")
    parser.add_argument("--old", required=True, help="旧物理集合名，如 kb_chunks_v1")
    parser.add_argument("--new", required=True, help="新物理集合名，如 kb_chunks_v2")
    parser.add_argument("--mode", default=index_config.default_rebuild_mode,
                        choices=["move_vectors", "re_embed"], help="重建模式，决定 Jaccard 门槛")
    parser.add_argument("--samples", type=int, default=index_config.validate_sample_size, help="取样条数")
    parser.add_argument("--seed", type=int, default=42, help="取样随机种子")
    parser.add_argument("--ef-search", type=int, default=None, help="指定 HNSW ef（默认用策略表）")
    parser.add_argument("--require-dataset", action="store_true",
                        help="评测集为空时直接失败，不退化成用正文当查询")
    parser.add_argument("--json-out", default=None, help="结果另存为 JSON 的路径")
    args = parser.parse_args(argv)

    result = validate(args.old, args.new, args.mode, args.samples, args.seed,
                      args.require_dataset, args.ef_search)
    print("\n" + format_report(result))

    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n结果已保存: {path}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
