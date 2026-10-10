# eval/generate_dataset.py

"""评测集生成 CLI：从语料反向出题（LLM 生成 query + 原切片作为标准答案）。

用法示例：
    python -m eval.generate_dataset --scenario param_lookup --limit 20   # 先小批量跑通
    python -m eval.generate_dataset                                      # 全部场景补齐到配额
    python -m eval.generate_dataset --quota 200 --no-dedup               # 不做语义去重
"""

import argparse
import json
import random
import sys
from pathlib import Path

from eval.config import SCENARIOS, ScenarioSpec, dataset_path, eval_config
from eval.dataset import append_records, iter_existing_queries, load_records, new_record
from tool.logger import logger
from utils.llm_utils import get_llm_client
from utils.milvus_utils import get_milvus_client

# 出题提示词：给一段真实切片，让 LLM 扮演用户提问，标准答案就是这段切片
GENERATE_PROMPT = """你在为一个产品知识库的检索系统造评测集。下面是知识库里的一段真实资料，请扮演一个真实用户，针对这段资料提出一个问题。

【场景】{label}
【出题要求】{instruction}

【硬性约束】
1. 问题的答案必须能在这段资料里找到，不要问资料里没有的内容。
2. 不要照抄资料里的原句，要用用户自己的话问（可以口语、可以省略主语）。
3. 只提一个问题，不要一次问多件事，不要加"请问""你好"之类的客套。
4. 长度控制在 8~40 个字之间。
5. 如果这段资料信息量太少、根本无法出题，把 answerable 置为 false。

【资料元信息】
- 源文件：{file_title}
- 章节标题：{title}
- 商品名：{item_name}

【资料正文】
{content}

只输出 JSON，格式：{{"query": "用户的问题", "answerable": true}}"""

# 抽样时取回的字段（三元组 + 出题所需正文）
SAMPLE_FIELDS = ["chunk_id", "content", "title", "parent_title", "part", "file_title", "item_name"]


def sample_chunks(collection: str, limit: int, min_len: int, max_len: int,
                  exclude: set, seed: int = None) -> list[dict]:
    """
    从 Milvus 抽样切片：分页拉取后在本地洗牌，避免只取到集合头部
    :param collection: 集合名
    :param limit: 期望抽到的条数（会多拉一些以补偿长度过滤的损耗）
    :param min_len: 正文长度下限
    :param max_len: 正文长度上限
    :param exclude: 已出题的三元组集合，抽样时跳过
    :param seed: 随机种子，便于复现
    :return: 切片字典列表
    """
    client = get_milvus_client()
    rng = random.Random(seed)

    pulled = []
    offset = 0
    page = min(eval_config.sample_batch_size, max(limit * 3, 100))
    # 多翻几页凑够候选量；单页不足 page 说明已经到底
    while len(pulled) < limit * 3:
        batch = client.query(
            collection_name=collection,
            filter="",
            output_fields=SAMPLE_FIELDS,
            limit=page,
            offset=offset,
        )
        if not batch:
            break
        pulled.extend(batch)
        if len(batch) < page:
            break
        offset += page

    rng.shuffle(pulled)

    picked = []
    for chunk in pulled:
        content = chunk.get("content") or ""
        if not (min_len <= len(content) <= max_len):
            continue
        key = (chunk.get("file_title"), chunk.get("title"), int(chunk.get("part") or 0))
        if key in exclude:
            continue
        exclude.add(key)
        picked.append(chunk)
        if len(picked) >= limit:
            break

    logger.info(f"抽样完成：拉取 {len(pulled)} 条，符合长度/去重条件 {len(picked)} 条")
    return picked


def generate_query(spec: ScenarioSpec, chunk: dict) -> str:
    """
    调 LLM 为一条切片出题
    :return: 问题文本；资料无法出题或调用失败时返回空串
    """
    prompt = GENERATE_PROMPT.format(
        label=spec.label,
        instruction=spec.instruction,
        file_title=chunk.get("file_title") or "",
        title=chunk.get("title") or "",
        item_name=chunk.get("item_name") or "",
        # 正文过长会顶爆上下文，出题只需要足够定位答案的片段
        content=(chunk.get("content") or "")[:4000],
    )

    try:
        llm = get_llm_client(model=eval_config.gen_llm_model or None, json_mode=True)
        content = llm.invoke(prompt).content
        if content.startswith("```"):
            content = content.strip("`")
            if content.startswith("json"):
                content = content[4:]
        parsed = json.loads(content)
        if not parsed.get("answerable", True):
            return ""
        query = (parsed.get("query") or "").strip()
        return query if 4 <= len(query) <= 80 else ""
    except Exception as e:
        logger.error(f"出题失败（chunk_id={chunk.get('chunk_id')}）: {e}")
        return ""


def semantic_duplicates(queries: list[str], existing_vectors: list[list]) -> list[bool]:
    """
    语义去重：与已有问题余弦相似度过高说明问法重复，丢弃
    :param queries: 本批新生成的问题
    :param existing_vectors: 已有问题的单位向量
    :return: 与 queries 等长的布尔列表，True 表示重复
    """
    from utils.embedding_utils import generate_embeddings

    if not queries:
        return []
    dense = generate_embeddings(queries).get("dense") or []
    flags = []
    for vec in dense:
        unit = _unit(vec)
        dup = any(_cosine(unit, ev) > eval_config.dedup_cosine_threshold for ev in existing_vectors)
        flags.append(dup)
        if not dup:
            # 同批内部也要去重，否则一次生成 50 条相似问法会全部保留
            existing_vectors.append(unit)
    return flags


def _unit(vec) -> list:
    vec = list(vec)
    norm = sum(x * x for x in vec) ** 0.5
    return [x / norm for x in vec] if norm else vec


def _cosine(a: list, b: list) -> float:
    return sum(x * y for x, y in zip(a, b))


def load_existing_vectors(scenario: str) -> tuple[set, list[list]]:
    """
    载入某场景已有数据集的去重依据
    :return: (已有问题字面集合, 已有问题单位向量列表)
    """
    path = dataset_path(scenario)
    existing_queries = set(iter_existing_queries(path))
    vectors = []
    if existing_queries:
        try:
            from utils.embedding_utils import generate_embeddings
            texts = sorted(existing_queries)
            vectors = [_unit(v) for v in (generate_embeddings(texts).get("dense") or [])]
            logger.info(f"已加载 {len(vectors)} 条历史问题向量用于语义去重")
        except Exception as e:
            logger.warning(f"语义去重不可用（{e}），本次仅做字面去重")
    return existing_queries, vectors


def used_triples(scenario: str) -> set:
    """某场景已出题的三元组集合，避免同一切片被反复出题"""
    return {
        (r.get("gold_file_title"), r.get("gold_title"), int(r.get("gold_part") or 0))
        for r in load_records(dataset_path(scenario), include_rejected=True)
    }


def run_scenario(spec: ScenarioSpec, limit: int, collection: str, dedup: bool,
                 seed: int = None, dry_run: bool = False) -> int:
    """
    为单个场景生成一批评测样本
    :param spec: 场景规格
    :param limit: 本次最多生成多少条
    :param collection: 语料集合名
    :param dedup: 是否做语义去重
    :param seed: 随机种子
    :param dry_run: True 时只抽样出题不调用 LLM（用于验证抽样链路）
    :return: 实际写入条数
    """
    path = dataset_path(spec.name)
    existing_queries, existing_vectors = load_existing_vectors(spec.name) if dedup else (set(), [])
    exclude = used_triples(spec.name)
    logger.info(f"[{spec.label}] 已有 {len(exclude)} 条样本，本次目标 {limit} 条")

    chunks = sample_chunks(collection, limit, spec.min_content_len, spec.max_content_len, exclude, seed)
    if not chunks:
        logger.warning(f"[{spec.label}] 没有可出题的切片（检查长度区间 {spec.min_content_len}~{spec.max_content_len}）")
        return 0

    if dry_run:
        for chunk in chunks[:5]:
            logger.info(f"  候选切片: {chunk.get('file_title')} | {chunk.get('title')} | part={chunk.get('part')} | len={len(chunk.get('content') or '')}")
        logger.info(f"[{spec.label}] dry-run 抽样 {len(chunks)} 条，未调用 LLM")
        return 0

    records = []
    for idx, chunk in enumerate(chunks, start=1):
        query = generate_query(spec, chunk)
        if not query:
            continue
        if query in existing_queries:
            logger.info(f"  丢弃（字面重复）: {query}")
            continue
        records.append(new_record(spec.name, query, chunk, spec.expected_intent,
                                  eval_config.gen_llm_model or "default"))
        existing_queries.add(query)
        if idx % 10 == 0:
            logger.info(f"[{spec.label}] 进度 {idx}/{len(chunks)}，累计出题 {len(records)}")

    if dedup and records:
        flags = semantic_duplicates([r["query"] for r in records], existing_vectors)
        kept = [r for r, dup in zip(records, flags) if not dup]
        logger.info(f"[{spec.label}] 语义去重丢弃 {len(records) - len(kept)} 条")
        records = kept

    written = append_records(path, records)
    logger.info(f"[{spec.label}] 写入 {written} 条 → {path}")
    return written


def fill_to_quota(spec: ScenarioSpec, collection: str, dedup: bool, seed: int = None,
                  dry_run: bool = False) -> int:
    """把某场景补齐到 quota_per_scenario（断点续跑：已有多少就补多少）"""
    existing = len(load_records(dataset_path(spec.name), include_rejected=True))
    remaining = eval_config.quota_per_scenario - existing
    if remaining <= 0:
        logger.info(f"[{spec.label}] 已达配额（{existing}/{eval_config.quota_per_scenario}），跳过")
        return 0
    return run_scenario(spec, remaining, collection, dedup, seed, dry_run)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="生成召回评测集（LLM 从语料反向出题）")
    parser.add_argument("--scenario", choices=list(SCENARIOS), help="只跑指定场景，默认全部")
    parser.add_argument("--limit", type=int, help="本次每场景最多生成条数（覆盖配额逻辑）")
    parser.add_argument("--quota", type=int, help="覆盖每场景目标配额，默认取 eval_config.quota_per_scenario")
    parser.add_argument("--collection", help="语料集合名，默认取 .env 的 CHUNKS_COLLECTION")
    parser.add_argument("--no-dedup", action="store_true", help="关闭语义去重（省一次 embedding）")
    parser.add_argument("--seed", type=int, help="随机种子，便于复现抽样")
    parser.add_argument("--dry-run", action="store_true", help="只验证抽样链路，不调 LLM、不写文件")
    args = parser.parse_args(argv)

    if args.quota:
        eval_config.quota_per_scenario = args.quota

    from config.milvus_config import milvus_config
    collection = args.collection or milvus_config.chunks_collection

    specs = [SCENARIOS[args.scenario]] if args.scenario else list(SCENARIOS.values())
    total = 0
    for spec in specs:
        if args.limit:
            total += run_scenario(spec, args.limit, collection, not args.no_dedup, args.seed, args.dry_run)
        else:
            total += fill_to_quota(spec, collection, not args.no_dedup, args.seed, args.dry_run)

    logger.info(f"本次共生成 {total} 条评测样本")
    if not args.dry_run:
        for spec in specs:
            path = dataset_path(spec.name)
            count = len(load_records(path, include_rejected=True))
            logger.info(f"  {spec.label}: {count}/{eval_config.quota_per_scenario}  ({path})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
