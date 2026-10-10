# processor/index_manager/quality.py

"""索引质量代理指标：不看召回率也能判断「索引是不是该重建了」。

为什么要代理指标
----------------
真正的召回率要跑评测集，慢且有成本，不可能每次查询都算。而索引劣化是有物理信号的：

· **sealed segment 越来越多** —— Milvus 把新写入的数据先放在小段里，段多了查询要扫的
  段就多，延迟上升。正常靠后台 compaction 合并，但删改频繁时合并跟不上，需要重建来压实。
· **行数与注册表记录对不上** —— 说明有导入漏写/双写失败，数据一致性出问题。
· **索引类型与配置不符** —— 比如还是 AUTOINDEX，说明这批数据是改造前建的，没吃到调优参数。

P3 只做「采集 + 告警」，P5 的调度器再拿这些信号自动触发重建。
"""

import argparse
import json
import sys

from config.index_config import index_config
from config.milvus_config import milvus_config
from tool.logger import logger
from utils import index_registry_utils as registry
from utils.milvus_utils import count_rows, get_milvus_client

# ORM 连接别名。MilvusClient（新 API）没有 segment 信息接口，只能用 ORM 的 utility
_ORM_ALIAS = "kb_quality"
_orm_ready = False


def _orm_connect() -> bool:
    """
    建一次 ORM 连接（get_query_segment_info 只在 ORM 侧提供）
    与主流程用的 MilvusClient 各走各的，互不影响
    """
    global _orm_ready
    if _orm_ready:
        return True
    try:
        from pymilvus import connections
        connections.connect(alias=_ORM_ALIAS, uri=milvus_config.milvus_url)
        _orm_ready = True
        return True
    except Exception as e:
        logger.error(f"Milvus ORM 连接失败，拿不到 segment 信息: {e}")
        return False


def segment_stats(collection: str) -> dict:
    """
    统计集合的 segment 分布
    :return: {sealed, growing, rows, max_segment_rows, states{}}；拿不到返回 {}
    """
    if not _orm_connect():
        return {}
    try:
        from pymilvus import utility
        segments = utility.get_query_segment_info(collection, using=_ORM_ALIAS) or []
    except Exception as e:
        # 集合没载入时拿不到 segment 信息，这是正常情况
        logger.warning(f"读取集合[{collection}] segment 信息失败: {e}")
        return {}

    states = {}
    sealed = growing = rows = max_rows = 0
    for segment in segments:
        state = getattr(segment, "state_name", "unknown")
        states[state] = states.get(state, 0) + 1
        num_rows = int(getattr(segment, "num_rows", 0) or 0)
        rows += num_rows
        max_rows = max(max_rows, num_rows)
        if state.lower() == "sealed":
            sealed += 1
        elif state.lower() == "growing":
            growing += 1

    return {"sealed": sealed, "growing": growing, "rows": rows,
            "max_segment_rows": max_rows, "states": states,
            "segment_count": len(segments)}


def index_quality(collection: str = None) -> dict:
    """
    采集一个集合的索引质量指标
    :param collection: 物理集合名，默认取注册表里的 active 版本
    :return: 指标字典，含 assess 给出的告警列表
    """
    client = get_milvus_client()
    logical = index_config.logical_collection
    record = registry.get_by_physical(collection) if collection else registry.get_active(logical)
    collection = collection or (record or {}).get("physical") or logical

    quality = {
        "collection": collection,
        "logical": logical,
        "version": (record or {}).get("version"),
        "status": (record or {}).get("status"),
        "registry_rows": (record or {}).get("row_count"),
        "rows": -1,
        "segments": {},
        "dense_index": {},
        "sparse_index": {},
        "warnings": [],
    }

    # 连不上 Milvus 时不硬闯：否则会冒出一堆 AttributeError 当「索引告警」，掩盖真正的问题
    if not client:
        quality["warnings"] = ["Milvus 连接失败，无法采集索引质量指标"]
        return quality

    quality["rows"] = count_rows(client, collection)
    quality["segments"] = segment_stats(collection)

    for field, name in (("dense_index", "dense_vector_index"), ("sparse_index", "sparse_inverted_index")):
        try:
            quality[field] = client.describe_index(collection, name)
        except Exception as e:
            quality[field] = {"error": str(e)}

    quality["warnings"] = assess(quality)
    return quality


def assess(quality: dict) -> list[str]:
    """
    按阈值判定，返回告警文案列表（空表示健康）
    只判断「已知的劣化信号」，不做召回率推断——那是评测体系的事
    """
    warnings = []
    segments = quality.get("segments") or {}

    sealed = segments.get("sealed")
    if sealed and sealed > index_config.max_sealed_segments:
        warnings.append(f"sealed segment 数 {sealed} 超过阈值 {index_config.max_sealed_segments}，"
                        f"碎片化严重，建议重建压实")

    rows = quality.get("rows", -1)
    registry_rows = quality.get("registry_rows")
    if rows >= 0 and registry_rows:
        # 行数偏差超过 1% 才报：Milvus 统计有压缩延迟，小偏差不说明问题
        if abs(rows - registry_rows) / max(registry_rows, 1) > 0.01:
            warnings.append(f"实际行数 {rows} 与注册表记录 {registry_rows} 偏差超过 1%，"
                            f"可能有导入漏写或双写失败")

    dense = quality.get("dense_index") or {}
    index_type = dense.get("index_type") or dense.get("index")
    if index_type and index_type != index_config.dense_index.index_type:
        warnings.append(f"稠密索引类型是 {index_type}，配置要求 "
                        f"{index_config.dense_index.index_type}，这批数据没吃到调优参数")

    if dense.get("error"):
        warnings.append(f"读不到稠密索引信息: {dense['error']}")

    return warnings


def format_report(quality: dict) -> str:
    """排版成给人看的文本"""
    segments = quality.get("segments") or {}
    lines = [
        f"索引质量: {quality['collection']} (version={quality.get('version')}, "
        f"status={quality.get('status')})",
        f"  行数: {quality.get('rows')}（注册表记录 {quality.get('registry_rows')}）",
        f"  segment: sealed={segments.get('sealed')} growing={segments.get('growing')} "
        f"最大段行数={segments.get('max_segment_rows')} 分布={segments.get('states')}",
        f"  稠密索引: {quality.get('dense_index')}",
        f"  稀疏索引: {quality.get('sparse_index')}",
    ]
    # 状态标记一律用 ASCII：Windows 控制台默认 GBK 编码，打印对勾/警告这类符号会
    # UnicodeEncodeError 直接崩掉 CLI（重建成功后那句提示崩了 = 明明成功却返回非 0）
    if quality["warnings"]:
        lines.append("  [!] 告警:")
        lines.extend(f"    - {w}" for w in quality["warnings"])
    else:
        lines.append("  [OK] 未发现劣化信号")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="采集索引质量代理指标")
    parser.add_argument("--collection", help="物理集合名，默认注册表里的 active 版本")
    parser.add_argument("--json", action="store_true", help="输出 JSON 而不是排版文本")
    args = parser.parse_args(argv)

    quality = index_quality(args.collection)
    if args.json:
        print(json.dumps(quality, ensure_ascii=False, indent=2, default=str))
    else:
        print(format_report(quality))
    return 1 if quality["warnings"] else 0


if __name__ == "__main__":
    sys.exit(main())
