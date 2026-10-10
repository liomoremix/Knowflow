# processor/index_manager/benchmark.py

"""HNSW 参数网格搜索：用评测集把「召回率 vs 延迟」量出来，再定参数。

为什么不能拍脑袋定 M / efConstruction / ef
------------------------------------------
· M、efConstruction 是**建索引时**定的，改不了，只能重建集合。越大图质量越好，
  但内存和建索引时间也涨。
· ef（ef_search）是**查询时**定的，随时可改。越大越接近暴力搜索的召回，延迟也越高。
这三个参数的取舍没有通用答案，取决于语料规模和分布，只能实测。

两种模式
--------
--ef-only：不建集合，只在当前集合上扫 ef。便宜、几分钟出结果。
           注意：如果当前索引不是 HNSW（比如还是 AUTOINDEX），ef 根本不生效，
           这时每一行的指标会完全一样——脚本会把索引类型打出来提醒你。
--build  ：全网格。每个 (M, efConstruction) 组合建一个临时集合、把向量搬过去、再扫 ef，
           跑完删掉。贵，但这是一次性的调参工作，定完参写进 .env 就不用再跑。

输出：markdown 表 + JSON，落到 eval/results/。挑一行「召回达标里延迟最低」的组合，
把 M/efConstruction 写进 .env（HNSW_M / HNSW_EF_CONSTRUCTION），
把 ef 写进 config/intent_config.py 的各意图策略里。

用法
----
    python -m processor.index_manager.benchmark --ef-only --limit 50
    python -m processor.index_manager.benchmark --build --limit 50
    python -m processor.index_manager.benchmark --build --M 16 32 --ef-construction 256 --ef-search 64 128
"""

import argparse
import json
import sys
import time
from datetime import datetime

from config.index_config import index_config
from config.milvus_config import milvus_config
from eval.config import SCENARIOS, eval_config
from eval.metrics import evaluate_query, summarize
from eval.retrieve import resolve_strategy, retrieve_raw, to_ranked_docs
from eval.run_eval import load_samples
from processor.index_manager.rebuild import COPY_FIELDS, copy_row, iter_rows
from tool.logger import logger
from utils.embedding_utils import generate_embeddings
from utils.milvus_utils import count_rows, create_chunks_collection, get_milvus_client, get_vector_dim

# 临时集合前缀，跑完就删；名字里带参数便于中断后人工清理
BENCH_PREFIX = "bench"


# ==================== 样本准备 ====================

def prepare_samples(scenarios: list[str], limit: int) -> list[dict]:
    """
    加载评测样本并**预先算好查询向量**。
    网格搜索要对同一批查询跑几十次，每次重新 embedding 是纯浪费；
    而且线上本来就是意图节点算一次、检索节点复用，这里保持一致。
    """
    records = load_samples(scenarios, limit)
    if not records:
        return []

    samples = []
    for idx, record in enumerate(records, start=1):
        query = record["query"]
        embeddings = generate_embeddings([query])
        samples.append({
            "qid": record.get("qid"),
            "scenario": record["scenario"],
            "query": query,
            "gold": (record["gold_file_title"], record["gold_title"], int(record.get("gold_part") or 0)),
            "item_names": [record["gold_item_name"]] if record.get("gold_item_name") else None,
            "reuse": {"query_dense_vec": embeddings["dense"][0],
                      "query_sparse_vec": embeddings["sparse"][0]},
        })
        if idx % 20 == 0:
            logger.info(f"查询向量预计算 {idx}/{len(records)}")
    return samples


# ==================== 单次测量 ====================

def measure(collection: str, samples: list[dict], ef_search: int, base_strategy: dict) -> dict:
    """在指定集合上按指定 ef 跑一遍评测，返回指标与延迟"""
    strategy = dict(base_strategy)
    strategy["ef_search"] = ef_search
    k_list = eval_config.k_list

    rows = []
    latencies = []
    errors = 0
    for sample in samples:
        try:
            started = time.perf_counter()
            chunks = retrieve_raw(sample["query"], strategy, sample["item_names"],
                                  collection=collection, reuse=sample["reuse"])
            latencies.append((time.perf_counter() - started) * 1000)
            rows.append({"scenario": sample["scenario"],
                         "scores": evaluate_query(sample["gold"], to_ranked_docs(chunks), k_list)})
        except Exception as e:
            errors += 1
            logger.error(f"检索失败 collection={collection} ef={ef_search}: {e}")

    summary = summarize(rows, k_list)
    ordered = sorted(latencies)
    return {
        "collection": collection,
        "ef_search": ef_search,
        "scored": len(rows),
        "errors": errors,
        "overall": summary.get("overall", {}),
        "latency_avg_ms": round(sum(ordered) / len(ordered), 1) if ordered else None,
        "latency_p50_ms": round(ordered[len(ordered) // 2], 1) if ordered else None,
        "latency_p95_ms": round(ordered[max(int(len(ordered) * 0.95) - 1, 0)], 1) if ordered else None,
    }


def describe_index(client, collection: str) -> dict:
    """读出集合当前的索引类型与参数（判断 ef 扫描有没有意义）"""
    info = {}
    for index_name in ("dense_vector_index", "sparse_inverted_index"):
        try:
            info[index_name] = client.describe_index(collection, index_name)
        except Exception as e:
            info[index_name] = {"error": str(e)}
    return info


# ==================== 建临时集合 ====================

def build_bench_collection(client, source: str, name: str, M: int, ef_construction: int,
                           batch_size: int) -> dict:
    """
    建一个用指定 HNSW 参数的临时集合并把向量搬过去
    :return: {rows, build_seconds, copy_seconds}
    """
    from dataclasses import replace

    dim = get_vector_dim(client, source)
    spec = replace(index_config.dense_index, M=M, ef_construction=ef_construction)

    started = time.perf_counter()
    if not create_chunks_collection(client, name, dim, spec):
        raise RuntimeError(f"临时集合 {name} 创建失败")
    build_seconds = time.perf_counter() - started

    total = count_rows(client, source)
    started = time.perf_counter()
    copied = 0
    buffer = []
    for row in iter_rows(client, source, COPY_FIELDS, batch_size):
        buffer.append(copy_row(row))
        if len(buffer) >= batch_size:
            client.insert(collection_name=name, data=buffer)
            copied += len(buffer)
            buffer = []
            logger.info(f"[{name}] 搬运 {copied}/{total}")
    if buffer:
        client.insert(collection_name=name, data=buffer)
        copied += len(buffer)
    copy_seconds = time.perf_counter() - started

    try:
        client.flush(name)
    except Exception as e:
        logger.warning(f"flush 失败: {e}")
    try:
        client.load_collection(name)
    except Exception as e:
        logger.warning(f"载入失败（可能已载入）: {e}")

    logger.info(f"[{name}] 建索引 {build_seconds:.1f}s，搬运 {copied} 行 {copy_seconds:.1f}s")
    return {"rows": copied, "build_seconds": round(build_seconds, 1),
            "copy_seconds": round(copy_seconds, 1)}


# ==================== 报告 ====================

def to_markdown(results: list[dict], gate_metric: str) -> str:
    """渲染成 markdown 表，直接贴进调参记录"""
    lines = [
        f"# HNSW 参数网格搜索  ({datetime.now().isoformat(timespec='seconds')})",
        "",
        f"门禁指标 `{gate_metric}`；每行是一次「建索引参数 + 查询参数」组合的实测结果。",
        "挑法：先筛掉召回不达标的行，再在剩下的里选 p95 延迟最低的。",
        "",
        "| 集合 | M | efConstruction | ef(search) | recall@5 | recall@10 | mrr | "
        "avg(ms) | p50(ms) | p95(ms) | 样本 | 建索引(s) |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in results:
        overall = row["overall"]
        lines.append(
            f"| {row['collection']} | {row.get('M', '-')} | {row.get('ef_construction', '-')} | "
            f"{row['ef_search']} | {overall.get('recall@5', 0):.4f} | "
            f"{overall.get('recall@10', 0):.4f} | {overall.get('mrr', 0):.4f} | "
            f"{row['latency_avg_ms']} | {row['latency_p50_ms']} | {row['latency_p95_ms']} | "
            f"{row['scored']} | {row.get('build_seconds', '-')} |"
        )
    return "\n".join(lines)


def save_reports(results: list[dict], meta: dict, tag: str) -> tuple[str, str]:
    """落 JSON + markdown，返回两个路径"""
    eval_config.result_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = eval_config.result_dir / f"{stamp}_bench_{tag}.json"
    md_path = eval_config.result_dir / f"{stamp}_bench_{tag}.md"

    json_path.write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, indent=2),
                         encoding="utf-8")
    md_path.write_text(to_markdown(results, eval_config.gate_metric), encoding="utf-8")
    return str(json_path), str(md_path)


# ==================== 主流程 ====================

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="HNSW 参数网格搜索")
    parser.add_argument("--build", action="store_true",
                        help="全网格：每个 (M, efConstruction) 建临时集合（贵，一次性调参用）")
    parser.add_argument("--ef-only", action="store_true",
                        help="只在当前集合上扫 ef_search，不建集合（默认）")
    parser.add_argument("--source", help="源集合名，默认 .env 的 CHUNKS_COLLECTION")
    parser.add_argument("--M", type=int, nargs="+", default=index_config.bench_M, help="M 取值列表")
    parser.add_argument("--ef-construction", type=int, nargs="+", default=index_config.bench_ef_construction,
                        help="efConstruction 取值列表")
    parser.add_argument("--ef-search", type=int, nargs="+", default=index_config.bench_ef_search,
                        help="查询期 ef 取值列表")
    parser.add_argument("--scenario", action="append", choices=list(SCENARIOS), help="只跑指定场景")
    parser.add_argument("--limit", type=int, default=50, help="每场景取样条数")
    parser.add_argument("--batch-size", type=int, default=index_config.backfill_batch_size, help="搬运批大小")
    parser.add_argument("--keep", action="store_true", help="保留临时集合（默认跑完就删）")
    parser.add_argument("--tag", default="grid", help="结果文件标签")
    args = parser.parse_args(argv)

    if args.build and args.ef_only:
        logger.error("--build 与 --ef-only 互斥")
        return 2

    source = args.source or milvus_config.chunks_collection

    client = get_milvus_client()
    if not client or not client.has_collection(source):
        logger.error(f"源集合 {source} 不可用")
        return 1

    logger.info(f"源集合 {source} 索引信息: {json.dumps(describe_index(client, source), ensure_ascii=False)}")

    scenarios = args.scenario or list(SCENARIOS)
    samples = prepare_samples(scenarios, args.limit)
    if not samples:
        logger.error("没有可用评测样本，请先运行 python -m eval.generate_dataset")
        return 2
    logger.info(f"评测样本 {len(samples)} 条，场景 {scenarios}")

    base_strategy = resolve_strategy()
    base_strategy["enable_hyde"] = False  # 只测索引本身，别把 HyDE 的大模型随机性混进来

    results = []
    built = []
    try:
        if args.build:
            for M in args.M:
                for ef_construction in args.ef_construction:
                    name = f"{BENCH_PREFIX}_{source}_M{M}_efc{ef_construction}"
                    logger.info(f"=== 建临时集合 {name} ===")
                    try:
                        build_info = build_bench_collection(client, source, name, M, ef_construction,
                                                            args.batch_size)
                    except Exception as e:
                        logger.error(f"{name} 建失败，跳过该组合: {e}")
                        continue
                    built.append(name)

                    for ef_search in args.ef_search:
                        row = measure(name, samples, ef_search, base_strategy)
                        row.update({"M": M, "ef_construction": ef_construction, **build_info})
                        results.append(row)
                        logger.info(f"[{name}] ef={ef_search} "
                                    f"recall@10={row['overall'].get('recall@10')} "
                                    f"p95={row['latency_p95_ms']}ms")
        else:
            if args.ef_only:
                logger.info("只扫 ef_search（不建集合）")
            for ef_search in args.ef_search:
                row = measure(source, samples, ef_search, base_strategy)
                results.append(row)
                logger.info(f"ef={ef_search} recall@10={row['overall'].get('recall@10')} "
                            f"p95={row['latency_p95_ms']}ms")
    finally:
        for name in built:
            if args.keep:
                logger.warning(f"--keep 已指定，临时集合 {name} 保留，记得人工 drop")
                continue
            try:
                client.drop_collection(name)
                logger.info(f"已删除临时集合 {name}")
            except Exception as e:
                logger.error(f"删除临时集合 {name} 失败，请人工处理: {e}")

    if not results:
        logger.error("没有任何测量结果")
        return 1

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": source,
        "mode": "build" if args.build else "ef_only",
        "scenarios": scenarios,
        "samples": len(samples),
        "grid": {"M": args.M, "ef_construction": args.ef_construction, "ef_search": args.ef_search},
        "source_index": describe_index(client, source),
        "eval_limits": {"per_path_limit": eval_config.eval_per_path_limit,
                        "hybrid_limit": eval_config.eval_hybrid_limit},
        "gate_metric": eval_config.gate_metric,
    }
    json_path, md_path = save_reports(results, meta, args.tag)

    print("\n" + to_markdown(results, eval_config.gate_metric))
    print(f"\nJSON: {json_path}\nMarkdown: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
