# processor/index_manager/migrate_alias.py

"""一次性迁移：把现有集合改造成「别名 + 版本化物理集合」结构。

为什么要迁移
------------
蓝绿重建的核心是「切换别名」：新旧两个物理集合都在，把 alias 一改，线上瞬间换索引。
但现在 .env 里的 CHUNKS_COLLECTION=kb_chunks 直接就是一个物理集合名，没有别名可切。

Milvus 里别名和集合名共用同一个命名空间，所以不能对 kb_chunks 直接 create_alias，
必须先把它改名：

    rename_collection("kb_chunks" -> "kb_chunks_v1")
    create_alias(collection_name="kb_chunks_v1", alias="kb_chunks")

迁移后 kb_chunks 从「集合」变成「指向 kb_chunks_v1 的别名」。查询侧配置一个字都不用改，
写入侧从索引注册表拿物理名（见 utils/index_registry_utils.get_write_targets）。

安全须知
--------
rename 期间 kb_chunks 这个名字短暂不存在，正在跑的查询会报错。
执行前先停掉 web/api 两个服务，迁移完再起。
默认只打印计划，加 --yes 才真正执行。

用法
----
    python -m processor.index_manager.migrate_alias                 # 只看计划
    python -m processor.index_manager.migrate_alias --yes           # 执行迁移
    python -m processor.index_manager.migrate_alias --yes --verify-dml  # 顺便验证别名可写
    python -m processor.index_manager.migrate_alias --rollback --yes    # 撤回迁移
"""

import argparse
import sys

from config.index_config import index_config
from tool.logger import logger
from utils.index_registry_utils import delete_version, register_existing
from utils.milvus_utils import (alias_target, get_milvus_client, get_row_count,
                               get_vector_dim)

# 验证别名可写时用的探针标记，插进去马上删掉
PROBE_FILE_TITLE = "__alias_probe__"


def inspect_state(client, logical: str, v1: str) -> dict:
    """
    探测当前处于迁移的哪个阶段（幂等与断点续跑都靠它判断）
    :return: {alias_target, logical_exists, v1_exists, action}
    """
    target = alias_target(client, logical)
    logical_exists = bool(client.has_collection(logical))
    v1_exists = bool(client.has_collection(v1))

    if target == v1:
        action = "done"            # 已迁移完成
    elif target:
        action = "conflict"        # 别名指向别的版本，人工确认
    elif v1_exists and not logical_exists:
        action = "alias_only"      # 改过名但别名没建（上次执行中断）
    elif logical_exists and v1_exists:
        action = "conflict"        # 两个都在，不敢猜哪个是真的
    elif logical_exists:
        action = "migrate"         # 正常待迁移状态
    else:
        action = "empty"           # 什么都没有：全新部署，导入时会自动建 v1

    return {"alias_target": target, "logical_exists": logical_exists,
            "v1_exists": v1_exists, "action": action}


def verify_dml(client, logical: str) -> bool:
    """
    通过别名做一次「插入 → 查回 → 删除」探针，确认这个 Milvus 版本允许对别名做 DML。

    为什么要单独验：主流程的写入走的是物理名，不依赖别名 DML；但重建脚本、临时补数据
    都可能图省事直接写别名。先验一次，比出问题时再猜好。
    """
    dim = get_vector_dim(client, logical)
    if not dim:
        logger.error(f"读不到集合[{logical}]的向量维度，跳过 DML 验证")
        return False

    probe = {
        "content": "alias dml probe",
        "title": "probe",
        "parent_title": "probe",
        "part": 0,
        "file_title": PROBE_FILE_TITLE,
        "item_name": "probe",
        "dense_vector": [0.0] * dim,
        "sparse_vector": {1: 0.1},
    }
    try:
        client.insert(collection_name=logical, data=[probe])
        rows = client.query(collection_name=logical,
                            filter=f'file_title == "{PROBE_FILE_TITLE}"',
                            output_fields=["chunk_id"], limit=1)
        found = bool(rows)
        client.delete(collection_name=logical, filter=f'file_title == "{PROBE_FILE_TITLE}"')
        logger.info(f"别名 DML 验证{'通过' if found else '失败：插入后查不到'}")
        return found
    except Exception as e:
        logger.error(f"别名 DML 验证异常（不影响迁移，写入侧走物理名）: {e}")
        try:
            client.delete(collection_name=logical, filter=f'file_title == "{PROBE_FILE_TITLE}"')
        except Exception:
            pass
        return False


def do_migrate(client, logical: str, v1: str, state: dict, do_verify_dml: bool) -> int:
    """执行迁移；返回退出码"""
    if state["action"] == "done":
        logger.info(f"已迁移：别名 {logical} → {state['alias_target']}，无需重复操作")
        register_existing(logical, version=1, physical=v1, row_count=get_row_count(client, v1))
        return 0

    if state["action"] == "conflict":
        logger.error(f"状态冲突，不敢自动处理：alias_target={state['alias_target']!r} "
                     f"{logical}存在={state['logical_exists']} {v1}存在={state['v1_exists']}")
        return 2

    if state["action"] == "empty":
        logger.warning(f"集合 {logical} 不存在：全新部署不需要迁移，"
                       f"首次导入会直接建出 {v1}（届时再跑一次本脚本登记注册表即可）")
        return 2

    rows_before = get_row_count(client, logical)
    logger.info(f"开始迁移：{logical} → {v1}，当前 {rows_before} 行")

    if state["action"] == "migrate":
        # 载入状态下改名可能被拒，先释放；失败不致命（有的版本不需要）
        try:
            client.release_collection(logical)
        except Exception as e:
            logger.warning(f"释放集合失败（继续尝试改名）: {e}")

        client.rename_collection(logical, v1)
        logger.info(f"改名完成：{logical} → {v1}")

    client.create_alias(collection_name=v1, alias=logical)
    logger.info(f"别名创建完成：{logical} → {v1}")

    # 载入并校验：这一步不过就说明迁移没成功，必须回滚，否则线上查不到数据
    try:
        client.load_collection(v1)
    except Exception as e:
        logger.warning(f"载入集合失败（可能已载入）: {e}")

    target = alias_target(client, logical)
    if target != v1:
        logger.error(f"校验失败：别名 {logical} 实际指向 {target!r}，期望 {v1}")
        return 1

    rows_after = get_row_count(client, logical)
    if rows_before >= 0 and rows_after != rows_before:
        logger.error(f"校验失败：行数不一致（迁移前 {rows_before}，通过别名读到 {rows_after}）")
        return 1
    logger.info(f"校验通过：别名可读，行数 {rows_after}")

    if do_verify_dml:
        verify_dml(client, logical)

    register_existing(logical, version=1, physical=v1, row_count=rows_after)
    logger.info("迁移完成。现在可以启动服务；后续重建用 processor.index_manager.rebuild")
    return 0


def do_rollback(client, logical: str, v1: str, state: dict) -> int:
    """撤回迁移：删别名 → 改回原名 → 清掉注册表记录"""
    if state["alias_target"] != v1:
        logger.error(f"当前别名 {logical} 指向 {state['alias_target']!r}，不是 {v1}，"
                     f"撤回会切错版本，请先人工确认")
        return 2
    if not state["logical_exists"]:
        logger.error(f"别名 {logical} 不存在，无需撤回")
        return 2

    client.drop_alias(logical)
    logger.info(f"已删除别名 {logical}")
    client.rename_collection(v1, logical)
    logger.info(f"已改回原名：{v1} → {logical}")
    try:
        client.load_collection(logical)
    except Exception as e:
        logger.warning(f"载入集合失败（可能已载入）: {e}")

    # 记录必须删掉：留着会被当成可清理的旧版本，清理任务会去 drop 现在正在线上跑的集合
    delete_version(logical, 1)
    logger.info("已撤回迁移，回到改造前状态")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="把现有 Milvus 集合迁移成别名+版本化物理集合")
    parser.add_argument("--yes", action="store_true", help="真正执行（默认只打印计划）")
    parser.add_argument("--rollback", action="store_true", help="撤回迁移，回到改造前状态")
    parser.add_argument("--verify-dml", action="store_true", help="迁移后插一条探针数据验证别名可写")
    parser.add_argument("--logical", default=index_config.logical_collection, help="逻辑集合名/别名")
    args = parser.parse_args(argv)

    logical = args.logical
    v1 = index_config.physical_name(1)

    client = get_milvus_client()
    if not client:
        logger.error("Milvus 连接失败")
        return 1

    state = inspect_state(client, logical, v1)
    logger.info(f"当前状态: {state}")

    if not args.yes:
        action = "撤回迁移" if args.rollback else "迁移"
        print(f"\n[{action}计划] 别名 {logical} → 物理集合 {v1}")
        print(f"  现在: alias_target={state['alias_target']!r} "
              f"{logical}存在={state['logical_exists']} {v1}存在={state['v1_exists']} "
              f"判定={state['action']}")
        print("  [!] 执行前先停掉 web/api 服务：rename 期间这个名字短暂不存在，查询会报错")
        print("  确认无误后加 --yes 执行")
        return 0

    logger.warning("执行期间请确保 web/api 服务已停止")
    if args.rollback:
        return do_rollback(client, logical, v1, state)
    return do_migrate(client, logical, v1, state, args.verify_dml)


if __name__ == "__main__":
    sys.exit(main())
