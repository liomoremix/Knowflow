# processor/index_manager/rebuild.py

"""蓝绿双索引重建：建新集合 → 双写 → 搬存量 → 校验 → 门禁 → 原子切换 → 旧版本排水。

为什么要这么绕
--------------
直接 drop 掉旧集合重建，期间知识库是空的，线上问答全部答不上来；重建到一半崩了，
数据就丢一半。蓝绿的做法是新集合在旁边建好、灌满、验过，最后只改一次别名指向，
切换是原子的、瞬间完成的，出问题一条命令切回去。

数据一致性怎么保证
------------------
关键在顺序：**先开双写，再搬存量**，而且搬运按 file_title 分组、搬之前先删新集合里的同名文件。

    t0  建好空的新集合
    t1  注册表登记 → 双写生效（此后每次导入都同时写新旧两个集合）
    t2  按 file_title 分组搬旧数据：先 delete 新集合里的该文件，再从旧集合整份拷过来

这样两种交错情况都是对的：
  · 文件 X 在 t2 之前被重新导入 → 双写已把新版写进新集合，t2 搬到 X 时先删再拷，
    拷的是旧集合里的同一份新数据，结果一致。
  · 文件 X 在 t2 之后被重新导入 → 双写在新集合里删掉搬过来的旧版、写入新版，结果一致。
所以既不会因为「搬完又被导入」而重复，也不会因为「导入完才被搬」而覆盖成旧数据。

切换前还要过两道闸门：validate.py 的一致性校验（行数/命中率/Jaccard/双写可见性）
和 eval.gate 的召回率门禁（跌幅超过 2% 直接拦下）。

用法
----
    python -m processor.index_manager.rebuild --status
    python -m processor.index_manager.rebuild                       # 只看计划
    python -m processor.index_manager.rebuild --yes                 # 全流程（含切换）
    python -m processor.index_manager.rebuild --yes --prepare-only  # 建好验好，停在 ready 等人工切
    python -m processor.index_manager.rebuild --switch --version 2 --yes
    python -m processor.index_manager.rebuild --rollback --version 1 --yes
    python -m processor.index_manager.rebuild --cleanup --yes       # 清理过了排水期的旧版本
"""

import argparse
import json
import sys
import time
from argparse import Namespace
from dataclasses import asdict, replace

from config.index_config import DenseIndexSpec, index_config
from eval import gate as eval_gate
from eval import run_eval
from eval.config import SCENARIOS
from processor.index_manager.validate import format_report, validate
from tool.logger import logger
from utils import index_registry_utils as registry
from utils.mongo_ops_utils import mongo_available
from utils.milvus_utils import (alias_target, count_rows, create_chunks_collection,
                                escape_milvus_string, get_milvus_client, get_vector_dim)

# 搬运时要带上的字段（chunk_id 是新集合自增的，不能搬）
COPY_FIELDS = ["content", "title", "parent_title", "part", "file_title", "item_name",
               "sparse_vector", "dense_vector"]
# re_embed 模式下只需要正文和元数据，向量重算
RE_EMBED_FIELDS = ["content", "title", "parent_title", "part", "file_title", "item_name"]
# 一次 embedding 多少条（BGE-M3 显存/内存占用不小，别太大）
EMBED_BATCH = 16
# 门禁评测默认每场景取多少条（子集，控成本）
DEFAULT_EVAL_LIMIT = 20


# ==================== 前置检查 ====================

def preflight(client, logical: str) -> tuple[dict, str]:
    """
    检查能不能开始重建
    :return: (active 记录, 错误信息)；错误信息非空表示不能继续
    """
    if not mongo_available():
        return None, "索引注册表不可用（Mongo 未连接）。重建进度必须跨进程可见，请先启动 Mongo。"

    target = alias_target(client, logical)
    if not target:
        return None, (f"别名 {logical} 不存在，还没做迁移。"
                      f"先跑 python -m processor.index_manager.migrate_alias --yes")

    active = registry.get_active(logical)
    if not active:
        return None, f"注册表里没有 active 版本（别名指向 {target}），请先跑 migrate_alias 登记。"

    if active.get("physical") != target:
        return None, (f"注册表与 Milvus 不一致：注册表 active={active.get('physical')}，"
                      f"别名实际指向 {target}。请先人工对齐，别在这种情况下重建。")

    building = registry.get_in_progress(logical)
    if building:
        return None, (f"已有在建版本 {building.get('physical')}（状态 {building.get('status')}）。"
                      f"先完成它，或用 --mark-failed --version {building.get('version')} 放弃。")

    if not client.has_collection(active["physical"]):
        return None, f"旧集合 {active['physical']} 在 Milvus 里不存在，注册表信息已过期。"

    return active, ""


# ==================== 搬运 ====================

def iter_rows(client, collection: str, fields: list[str], batch_size: int, extra_filter: str = ""):
    """
    按 chunk_id 游标分页拉取
    为什么不用 offset 分页：offset 越大 Milvus 要跳过的数据越多，深分页会越来越慢，
    而且 offset+limit 有上限。用「上次见过的最大 id」当游标就没这些问题。
    """
    cursor = 0
    while True:
        condition = f"chunk_id > {cursor}"
        if extra_filter:
            condition = f"{extra_filter} and {condition}"
        rows = client.query(collection_name=collection, filter=condition,
                            output_fields=["chunk_id"] + fields, limit=batch_size)
        if not rows:
            return
        for row in rows:
            yield row
        cursor = max(int(row["chunk_id"]) for row in rows)
        if len(rows) < batch_size:
            return


def _list_file_titles(client, collection: str, batch_size: int) -> list[str]:
    """列出集合里所有 file_title（搬运按文件分组，才能做到「先删后拷」的幂等）"""
    titles = set()
    for row in iter_rows(client, collection, ["file_title"], batch_size * 5):
        if row.get("file_title"):
            titles.add(row["file_title"])
    return sorted(titles)


def _normalize_sparse(value):
    """Milvus 返回的稀疏向量可能是 dict 之外的映射类型，统一成 dict 才能再插回去"""
    if value is None or isinstance(value, dict):
        return value
    try:
        return dict(value)
    except Exception:
        return value


def copy_row(row: dict) -> dict:
    """把旧集合的一行转成新集合可插入的格式（去掉主键，让新集合自己生成）"""
    data = {field: row.get(field) for field in COPY_FIELDS if field in row}
    data["sparse_vector"] = _normalize_sparse(data.get("sparse_vector"))
    if "part" not in data:
        data["part"] = 0
    return data


def backfill_move(client, old: str, new: str, logical: str, version: int,
                  batch_size: int, total: int) -> int:
    """
    move_vectors：直接把旧集合的向量搬过去（快、不花模型算力，前提是 embedding 模型没变）
    :return: 实际搬运的行数
    """
    titles = _list_file_titles(client, old, batch_size)
    logger.info(f"待搬运文件数: {len(titles)}")
    copied = 0

    for idx, title in enumerate(titles, start=1):
        condition = f"file_title == '{escape_milvus_string(title)}'"
        # 先删再拷：让搬运变成幂等操作，重建期间该文件被重新导入也不会留下重复数据
        client.delete(collection_name=new, filter=condition)

        buffer = []
        for row in iter_rows(client, old, COPY_FIELDS, batch_size, extra_filter=condition):
            buffer.append(copy_row(row))
            if len(buffer) >= batch_size:
                client.insert(collection_name=new, data=buffer)
                copied += len(buffer)
                buffer = []
                registry.update_progress(logical, version, copied, total)
        if buffer:
            client.insert(collection_name=new, data=buffer)
            copied += len(buffer)

        registry.update_progress(logical, version, copied, total)
        logger.info(f"搬运进度 {idx}/{len(titles)} 文件 | 已搬 {copied}/{total} 行 | {title}")

    return copied


def backfill_re_embed(client, old: str, new: str, logical: str, version: int,
                      batch_size: int, total: int) -> int:
    """
    re_embed：读旧集合的正文重新算向量（换了 embedding 模型才用，慢且吃算力）
    :return: 实际写入的行数
    """
    from utils.embedding_utils import generate_embeddings

    titles = _list_file_titles(client, old, batch_size)
    logger.info(f"待重算文件数: {len(titles)}")
    copied = 0

    for idx, title in enumerate(titles, start=1):
        condition = f"file_title == '{escape_milvus_string(title)}'"
        client.delete(collection_name=new, filter=condition)

        rows = list(iter_rows(client, old, RE_EMBED_FIELDS, batch_size, extra_filter=condition))
        for start in range(0, len(rows), EMBED_BATCH):
            batch = rows[start:start + EMBED_BATCH]
            embeddings = generate_embeddings([row.get("content") or "" for row in batch])
            data = []
            for offset, row in enumerate(batch):
                item = {field: row.get(field) for field in RE_EMBED_FIELDS if field in row}
                item["part"] = item.get("part", 0)
                item["dense_vector"] = embeddings["dense"][offset]
                item["sparse_vector"] = embeddings["sparse"][offset]
                data.append(item)
            client.insert(collection_name=new, data=data)
            copied += len(data)
            registry.update_progress(logical, version, copied, total)

        logger.info(f"重算进度 {idx}/{len(titles)} 文件 | 已写 {copied}/{total} 行 | {title}")

    return copied


# ==================== 门禁评测 ====================

def _eval_args(collection: str, tag: str, no_rerank: bool) -> Namespace:
    """
    拼一个 run_eval 需要的参数对象
    run_eval 是 CLI，这里复用它的 run/build_result，不重新实现评测逻辑
    """
    return Namespace(
        tag=tag, version=None, collection=collection, scenario=None, limit=None,
        intent=None, use_intent=False, ef_search=None, dense_weight=None, sparse_weight=None,
        hyde=None, no_rerank=no_rerank, no_item_filter=False, no_details=True, output=None,
    )


def run_eval_on(collection: str, tag: str, limit: int, scenarios: list[str],
                no_rerank: bool = True) -> str:
    """在指定集合上跑一次评测子集，返回结果文件路径"""
    samples = run_eval.load_samples(scenarios, limit)
    if not samples:
        raise RuntimeError("评测集为空，无法做门禁。先跑 python -m eval.generate_dataset")

    args = _eval_args(collection, tag, no_rerank)
    rows, stats = run_eval.run(samples, args)
    if stats["error_rate"] > run_eval.MAX_ERROR_RATE:
        raise RuntimeError(f"评测检索失败率 {stats['error_rate']:.1%} 过高，指标不可信")

    result = run_eval.build_result(rows, stats, args, scenarios)
    path = run_eval.result_path(tag)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info(f"[{collection}] 门禁评测完成: {path} (样本 {stats['scored']}/{stats['total_samples']})")
    return path


def run_gate(old: str, new: str, limit: int, scenarios: list[str], no_rerank: bool) -> tuple[bool, str]:
    """
    新旧集合各跑一次评测，用 P2 的门禁比较
    :return: (是否通过, 候选结果文件路径)
    """
    stamp = time.strftime("%Y%m%d_%H%M%S")
    base_path = run_eval_on(old, f"{stamp}_gate_old", limit, scenarios, no_rerank)
    cand_path = run_eval_on(new, f"{stamp}_gate_new", limit, scenarios, no_rerank)
    code = eval_gate.main(["--baseline", base_path, "--candidate", cand_path])
    return code == 0, cand_path


# ==================== 主流程 ====================

def switch_alias(client, logical: str, physical: str) -> bool:
    """
    原子切换别名 → 新集合。这一步是瞬时生效的，线上查询立刻打到新索引。
    :return: 是否切换成功并已校验
    """
    client.alter_alias(collection_name=physical, alias=logical)
    target = alias_target(client, logical)
    if target != physical:
        logger.error(f"切换后校验失败：别名 {logical} 指向 {target!r}，期望 {physical}")
        return False
    logger.info(f"别名已切换: {logical} → {physical}")
    return True


def run_rebuild(client, logical: str, args) -> int:
    """完整重建流程；返回退出码"""
    active, error = preflight(client, logical)
    if error:
        logger.error(error)
        return 2

    old = active["physical"]
    mode = args.mode
    batch_size = args.batch_size or index_config.backfill_batch_size
    dense_index = _dense_index(args)

    old_rows = count_rows(client, old)
    if old_rows < 0:
        logger.error(f"读不到旧集合 {old} 的行数（集合可能未载入），无法重建")
        return 1
    dim = get_vector_dim(client, old)
    if not dim:
        logger.error(f"读不到旧集合 {old} 的向量维度，无法建新集合")
        return 1

    version = registry.next_version(logical)
    new = args.new_collection or index_config.physical_name(version)

    print(f"\n[重建计划]")
    print(f"  旧集合: {old} ({old_rows} 行)")
    print(f"  新集合: {new} (version={version})")
    print(f"  模式:   {mode}")
    print(f"  稠密索引: {dense_index.index_type} M={dense_index.M} efConstruction={dense_index.ef_construction}")
    print(f"  稀疏索引: {index_config.sparse_index.index_type}")
    print(f"  双写:   {'开' if index_config.double_write_enabled else '关'}")
    print(f"  闸门:   一致性校验({index_config.validate_sample_size} 样本) + 召回门禁"
          f"(每场景 {args.eval_limit} 条, {eval_gate.eval_config.gate_metric} 跌幅 ≤ "
          f"{eval_gate.eval_config.gate_recall_drop:.0%})")
    print(f"  切换:   {'否（停在 ready）' if args.prepare_only else '是（alter_alias）'}")

    if not args.yes:
        print("\n这是只读预览。确认后加 --yes 执行。")
        print("建议先停掉导入任务；查询服务可以不停（切换是原子的）。")
        return 0

    if client.has_collection(new):
        logger.error(f"集合 {new} 已存在，换个名字（--new-collection）或先手动 drop")
        return 2

    # 先建空集合，再登记双写。
    # 反过来的话有个竞态：登记一生效双写就开了，此时若有导入进来而新集合还不存在，
    # 导入节点会按配置里的默认参数把它建出来，CLI 指定的 --M/--ef-construction 就白传了。
    if not create_chunks_collection(client, new, dim, dense_index):
        logger.error(f"新集合 {new} 创建失败")
        return 1

    record = registry.register_build(logical, mode=mode, total=old_rows, physical=new,
                                     version=version, dense_index=asdict(dense_index),
                                     sparse_index=index_config.sparse_index_dict())
    if not record:
        logger.error("登记重建任务失败，删除刚建好的空集合")
        client.drop_collection(new)
        return 2
    version = record["version"]
    new = record["physical"]

    try:
        # --- 1. 搬存量（此时双写已生效，重建期间的新导入会同时落到两个集合） ---
        registry.set_status(logical, version, registry.STATUS_BACKFILLING)
        started = time.time()
        backfill = backfill_move if mode == "move_vectors" else backfill_re_embed
        copied = backfill(client, old, new, logical, version, batch_size, old_rows)
        logger.info(f"存量搬运完成: {copied} 行，耗时 {time.time() - started:.1f}s")

        try:
            client.flush(new)
        except Exception as e:
            logger.warning(f"flush 失败（不影响后续，计数可能略有延迟）: {e}")
        try:
            client.load_collection(new)
        except Exception as e:
            logger.warning(f"载入新集合失败（可能已载入）: {e}")

        # --- 2. 一致性校验 ---
        registry.set_status(logical, version, registry.STATUS_VALIDATING, row_count=copied)
        result = validate(old, new, mode=mode, samples=args.validate_samples)
        print("\n" + format_report(result))
        registry.set_status(logical, version, registry.STATUS_VALIDATING, validation=result)
        if not result["passed"]:
            raise RuntimeError("一致性校验未通过，细节见上方报告")

        # --- 3. 召回门禁 ---
        if args.skip_eval:
            logger.warning("--skip-eval 已指定，跳过召回门禁（不建议：换索引参数可能掉召回）")
            eval_ref = ""
        else:
            scenarios = args.scenario or list(SCENARIOS)
            passed, eval_ref = run_gate(old, new, args.eval_limit, scenarios,
                                        no_rerank=not args.eval_full_chain)
            if not passed:
                raise RuntimeError("召回门禁拦截：新索引召回率跌幅超过阈值")
            registry.set_status(logical, version, registry.STATUS_VALIDATING, eval_result_ref=eval_ref)

        # --- 4. 就绪 ---
        registry.set_status(logical, version, registry.STATUS_READY,
                            row_count=count_rows(client, new), eval_result_ref=eval_ref)
        logger.info(f"{new} 已就绪（状态 ready），随时可切换")

        if args.prepare_only:
            print(f"\n已停在 ready。人工确认后切换：")
            print(f"  python -m processor.index_manager.rebuild --switch --version {version} --yes")
            return 0

        # --- 5. 原子切换：先改别名，成功了再改注册表 ---
        if not switch_alias(client, logical, new):
            raise RuntimeError("别名切换失败，注册表未变更，线上仍指向旧集合")
        if not registry.activate(logical, version):
            logger.error("别名已切换但注册表更新失败，请人工修正注册表！"
                         f"（当前线上指向 {new}）")
            return 1

        print(f"\n[OK] 重建完成并已切换: {logical} → {new}")
        print(f"   旧集合 {old} 进入排水期，{index_config.retire_after_days} 天后可清理："
              f"python -m processor.index_manager.rebuild --cleanup --yes")
        print(f"   需要回滚：python -m processor.index_manager.rebuild --rollback "
              f"--version {active['version']} --yes")
        return 0

    except Exception as e:
        logger.error(f"重建失败: {e}", exc_info=True)
        registry.mark_failed(logical, version, str(e))
        print(f"\n[FAIL] 重建失败，{new} 已标记为 failed，线上仍指向 {old}（未受影响）")
        if args.drop_on_failure:
            client.drop_collection(new)
            logger.info(f"已删除失败的新集合 {new}")
        else:
            print(f"   排查后可手动删除: 在 Milvus 里 drop_collection('{new}')，"
                  f"或加 --drop-on-failure 重跑")
        return 1


def _dense_index(args) -> DenseIndexSpec:
    """按 CLI 覆盖参数生成稠密索引规格（网格搜索定完参就用这里传进来）"""
    spec = index_config.dense_index
    if args.M or args.ef_construction or args.index_type:
        spec = replace(spec,
                       M=args.M or spec.M,
                       ef_construction=args.ef_construction or spec.ef_construction,
                       index_type=args.index_type or spec.index_type)
    return spec


# ==================== 其它子命令 ====================

def do_switch(client, logical: str, version: int) -> int:
    """把别名切到指定版本（prepare-only 之后的人工确认步骤，也是回滚用的动作）"""
    record = registry.get(logical, version)
    if not record:
        logger.error(f"注册表里没有 logical={logical} version={version}")
        return 2
    physical = record["physical"]
    if not client.has_collection(physical):
        logger.error(f"集合 {physical} 不存在，无法切换")
        return 2
    if record["status"] not in (registry.STATUS_READY, registry.STATUS_ACTIVE, registry.STATUS_DRAINING):
        logger.error(f"版本 {physical} 状态是 {record['status']}，只有 ready/active/draining 才允许切换")
        return 2

    if not switch_alias(client, logical, physical):
        return 1
    if not registry.activate(logical, version):
        logger.error("别名已切换但注册表更新失败，请人工修正注册表！")
        return 1
    print(f"[OK] 已切换: {logical} → {physical} (version={version})")
    return 0


def do_cleanup(client, logical: str, dry: bool) -> int:
    """清理过了排水期的旧版本：删物理集合 + 注册表标记 retired"""
    retirable = registry.list_retirable(logical)
    if not retirable:
        print(f"没有可清理的版本（排水期 {index_config.retire_after_days} 天）")
        return 0

    for record in retirable:
        physical = record["physical"]
        if alias_target(client, logical) == physical:
            logger.warning(f"{physical} 仍是别名指向的集合，跳过（注册表状态可能不准）")
            continue
        print(f"  待清理: {physical} (version={record['version']}, 状态 {record['status']})")
        if dry:
            continue
        client.drop_collection(physical)
        registry.retire(logical, record["version"], dropped=True)
        logger.info(f"已清理 {physical}")

    if dry:
        print("\n这是预览，加 --yes 真正删除")
    return 0


def do_mark_failed(logical: str, version: int, reason: str) -> int:
    """把一个卡住的在建版本标记为 failed，退出双写，让新的重建能开始"""
    record = registry.get(logical, version)
    if not record:
        logger.error(f"注册表里没有 logical={logical} version={version}")
        return 2
    registry.mark_failed(logical, version, reason)
    print(f"已标记 {record['physical']} 为 failed（原因: {reason}）；集合需人工 drop")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="蓝绿双索引重建")
    parser.add_argument("--logical", default=index_config.logical_collection, help="逻辑集合名/别名")
    parser.add_argument("--yes", action="store_true", help="真正执行（默认只打印计划）")

    parser.add_argument("--status", action="store_true", help="打印注册表状态后退出")
    parser.add_argument("--switch", action="store_true", help="只切换别名到 --version 指定的版本")
    parser.add_argument("--rollback", action="store_true", help="回滚到 --version 指定的版本（等价于 switch）")
    parser.add_argument("--cleanup", action="store_true", help="清理过了排水期的旧版本")
    parser.add_argument("--mark-failed", action="store_true", help="把卡住的在建版本标记为 failed")
    parser.add_argument("--version", type=int, help="配合 --switch/--rollback/--mark-failed 使用")

    parser.add_argument("--mode", default=index_config.default_rebuild_mode,
                        choices=["move_vectors", "re_embed"], help="搬向量还是重新算向量")
    parser.add_argument("--new-collection", help="新集合名，默认 {logical}_v{下一个版本号}")
    parser.add_argument("--index-type", help="覆盖稠密索引类型，默认 HNSW")
    parser.add_argument("--M", type=int, help="覆盖 HNSW 的 M")
    parser.add_argument("--ef-construction", type=int, help="覆盖 HNSW 的 efConstruction")
    parser.add_argument("--batch-size", type=int, help="搬运批大小")

    parser.add_argument("--prepare-only", action="store_true", help="建好验好停在 ready，不切别名")
    parser.add_argument("--validate-samples", type=int, help="一致性校验取样条数")
    parser.add_argument("--skip-eval", action="store_true", help="跳过召回门禁（不建议）")
    parser.add_argument("--scenario", action="append", choices=list(SCENARIOS), help="门禁只跑指定场景")
    parser.add_argument("--eval-limit", type=int, default=DEFAULT_EVAL_LIMIT, help="门禁每场景取样条数")
    parser.add_argument("--eval-full-chain", action="store_true",
                        help="门禁评测走完整链路（含 rerank，慢且要调外部服务），默认只测召回层")
    parser.add_argument("--drop-on-failure", action="store_true", help="失败时自动删除新集合")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logical = args.logical

    if args.status:
        print(json.dumps(registry.registry_status(logical), ensure_ascii=False, indent=2))
        return 0

    client = get_milvus_client()
    if not client:
        logger.error("Milvus 连接失败")
        return 1

    if args.mark_failed:
        if not args.version:
            logger.error("--mark-failed 需要 --version")
            return 2
        return do_mark_failed(logical, args.version, "人工标记失败")

    if args.switch or args.rollback:
        if not args.version:
            logger.error("--switch/--rollback 需要 --version")
            return 2
        if not args.yes:
            record = registry.get(logical, args.version)
            print(f"将把别名 {logical} 切到 "
                  f"{(record or {}).get('physical', f'version={args.version}')}（加 --yes 执行）")
            return 0
        return do_switch(client, logical, args.version)

    if args.cleanup:
        return do_cleanup(client, logical, dry=not args.yes)

    return run_rebuild(client, logical, args)


if __name__ == "__main__":
    sys.exit(main())
