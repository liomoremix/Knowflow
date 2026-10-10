# eval/dataset.py

"""评测数据集的读写与「业务三元组」定义。

为什么不用 chunk_id 做标准答案：chunk_id 是 Milvus 自增主键，索引重建（P3 蓝绿切换）后
同一段文字会拿到全新的 id，评测集就会整体失效。因此命中判定统一用业务三元组
(file_title, title, part)，另存 content_hash 用于检测语料本身是否被改过。
"""

import hashlib
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

# 数据集记录的字段顺序（写文件时按此顺序，方便人工阅读）
FIELDS = [
    "qid", "scenario", "query",
    "gold_file_title", "gold_title", "gold_part", "gold_item_name",
    "gold_content_hash", "gold_content", "gold_chunk_id_at_gen",
    "expected_intent", "gen_model", "verified", "created_at",
]


def content_hash(content: str) -> str:
    """切片正文的内容指纹，用于判断语料是否发生变化"""
    return hashlib.sha256((content or "").encode("utf-8")).hexdigest()[:16]


def gold_key(record: dict) -> tuple:
    """一条数据集记录的标准答案键（业务三元组）"""
    return (
        record.get("gold_file_title"),
        record.get("gold_title"),
        int(record.get("gold_part") or 0),
    )


def hit_key(doc: dict) -> tuple:
    """一条检索结果的命中键，与 gold_key 同构，可直接比较"""
    return (
        doc.get("file_title"),
        doc.get("title"),
        int(doc.get("part") or 0),
    )


def new_record(
        scenario: str,
        query: str,
        chunk: dict,
        expected_intent: str = "",
        gen_model: str = "",
) -> dict:
    """由「一条切片 + LLM 生成的问题」组装一条数据集记录"""
    content = chunk.get("content") or ""
    return {
        "qid": f"{scenario}-{uuid.uuid4().hex[:12]}",
        "scenario": scenario,
        "query": query,
        "gold_file_title": chunk.get("file_title"),
        "gold_title": chunk.get("title"),
        "gold_part": int(chunk.get("part") or 0),
        "gold_item_name": chunk.get("item_name"),
        "gold_content_hash": content_hash(content),
        "gold_content": content,
        "gold_chunk_id_at_gen": str(chunk.get("chunk_id") or ""),
        "expected_intent": expected_intent,
        "gen_model": gen_model,
        # None = 未抽查；True = 人工确认；False = 人工驳回（不计入评测）
        "verified": None,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def append_records(path: Path, records: Iterable[dict]) -> int:
    """追加写入 jsonl，返回写入条数"""
    records = list(records)
    if not records:
        return 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            ordered = {k: record.get(k) for k in FIELDS}
            f.write(json.dumps(ordered, ensure_ascii=False) + "\n")
    return len(records)


def load_records(path: Path, include_rejected: bool = False) -> list[dict]:
    """读取 jsonl 数据集；默认剔除人工驳回(verified=False)的记录"""
    if not Path(path).exists():
        return []
    records = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not include_rejected and record.get("verified") is False:
                continue
            records.append(record)
    return records


def rewrite_records(path: Path, records: Iterable[dict]) -> int:
    """整体重写 jsonl（人工抽查回写 verified 标记时用），返回条数"""
    records = list(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            ordered = {k: record.get(k) for k in FIELDS}
            # 保留 FIELDS 之外的自定义字段，避免回写时丢数据
            ordered.update({k: v for k, v in record.items() if k not in FIELDS})
            f.write(json.dumps(ordered, ensure_ascii=False) + "\n")
    return len(records)


def iter_existing_queries(path: Path) -> Iterator[str]:
    """流式读取已有问题文本（生成时做字面去重用）"""
    if not Path(path).exists():
        return
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line).get("query") or ""
