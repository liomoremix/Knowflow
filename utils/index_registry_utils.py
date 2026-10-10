# utils/index_registry_utils.py

"""索引注册表（Mongo collection: index_registry）——蓝绿双索引的状态机。

为什么需要它：
1. 重建进度必须跨进程可见。utils/task_utils.py 是进程内存字典，web 服务查不到 CLI 跑的进度。
2. 导入节点要知道「现在该往哪几个集合写」（单写 active，还是双写 active+building）。
3. 切换/回滚要有据可查：哪个版本、什么时候激活、评测结果是哪份文件。

状态机：
    building ──> backfilling ──> validating ──> ready ──> active
        │             │              │                        │
        └─────────────┴──────────────┴──> failed              └──> draining ──> retired
                                                        （切换后旧版本进入排水期，留回滚窗口）

Mongo 不可用时全部函数返回 None / 退化值，导入与查询主流程不受影响。
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from pymongo import ASCENDING, DESCENDING

from config.index_config import index_config
from tool.logger import logger
from utils.mongo_ops_utils import as_dict, get_collection

COLLECTION_NAME = "index_registry"

INDEXES = [
    # 同一逻辑集合的同一版本号唯一，防止并发重建撞号
    ([("logical", ASCENDING), ("version", DESCENDING)], {"unique": True}),
    ([("status", ASCENDING)], {}),
    ([("physical", ASCENDING)], {}),
]

# ---- 状态常量 ----
STATUS_BUILDING = "building"        # 新集合已建，正在灌数据
STATUS_BACKFILLING = "backfilling"  # 存量搬完，双写已开，正在补最后的增量
STATUS_VALIDATING = "validating"    # 正在做新旧集合一致性校验
STATUS_READY = "ready"              # 校验+门禁都过了，等待切换
STATUS_ACTIVE = "active"            # 当前线上正在用的版本（alias 指向它）
STATUS_DRAINING = "draining"        # 已被新版本替换，保留观察期，可回滚
STATUS_RETIRED = "retired"          # 观察期结束，物理集合已删
STATUS_FAILED = "failed"            # 构建/校验失败，不参与双写，等待人工处理

# 在建状态：这些状态下导入需要双写
IN_PROGRESS = [STATUS_BUILDING, STATUS_BACKFILLING, STATUS_VALIDATING, STATUS_READY]
# 终态：不再变化
TERMINAL = [STATUS_RETIRED, STATUS_FAILED]


def _coll():
    """取注册表集合；Mongo 不可用时返回 None"""
    return get_collection(COLLECTION_NAME, INDEXES)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


# ==================== 查询 ====================

def get_active(logical: str = None) -> Optional[dict]:
    """当前线上生效的版本（alias 指向的物理集合）"""
    coll = _coll()
    if coll is None:
        return None
    logical = logical or index_config.logical_collection
    return as_dict(coll.find_one({"logical": logical, "status": STATUS_ACTIVE}))


def get_in_progress(logical: str = None) -> Optional[dict]:
    """当前在建的版本；同时有多个属于异常状态，取最新的一个"""
    coll = _coll()
    if coll is None:
        return None
    logical = logical or index_config.logical_collection
    doc = coll.find_one(
        {"logical": logical, "status": {"$in": IN_PROGRESS}},
        sort=[("version", DESCENDING)],
    )
    return as_dict(doc)


def get(logical: str, version: int) -> Optional[dict]:
    """按逻辑名+版本号取记录"""
    coll = _coll()
    if coll is None:
        return None
    return as_dict(coll.find_one({"logical": logical, "version": version}))


def get_by_physical(physical: str) -> Optional[dict]:
    """按物理集合名反查记录"""
    coll = _coll()
    if coll is None:
        return None
    return as_dict(coll.find_one({"physical": physical}))


def list_versions(logical: str = None) -> List[dict]:
    """列出某逻辑集合的全部版本（新版本在前）"""
    coll = _coll()
    if coll is None:
        return []
    logical = logical or index_config.logical_collection
    return [as_dict(d) for d in coll.find({"logical": logical}).sort("version", DESCENDING)]


def next_version(logical: str = None) -> int:
    """下一个可用版本号（从 1 开始递增）"""
    coll = _coll()
    logical = logical or index_config.logical_collection
    if coll is None:
        return 1
    latest = coll.find_one({"logical": logical}, sort=[("version", DESCENDING)])
    return int((latest or {}).get("version", 0)) + 1


# ==================== 写入 ====================

def register_build(logical: str = None, mode: str = None, total: int = 0,
                   dense_index: dict = None, sparse_index: dict = None,
                   embedding_model: str = "", physical: str = None,
                   version: int = None) -> Optional[dict]:
    """
    登记一次重建：分配版本号，写入 building 记录
    同一逻辑集合同时只允许一个在建版本，否则返回 None（调用方应先处理掉旧的）
    :param mode: move_vectors / re_embed
    :param total: 待搬运/重算的总条数，用于进度百分比
    :return: 新记录；已有在建版本或 Mongo 不可用时返回 None
    """
    coll = _coll()
    if coll is None:
        logger.error("索引注册表不可用（Mongo 未连接），无法登记重建任务")
        return None

    logical = logical or index_config.logical_collection
    if get_in_progress(logical):
        existing = get_in_progress(logical)
        logger.error(f"已有在建版本 {existing.get('physical')}（状态 {existing.get('status')}），"
                     f"请先完成或标记失败后再发起新的重建")
        return None

    version = version or next_version(logical)
    doc = {
        "logical": logical,
        "alias": logical,  # alias 名与逻辑名一致，查询侧配置不用改
        "version": version,
        "physical": physical or index_config.physical_name(version),
        "status": STATUS_BUILDING,
        "mode": mode or index_config.default_rebuild_mode,
        "embedding_model": embedding_model,
        "dense_index": dense_index or index_config.dense_index_dict(),
        "sparse_index": sparse_index or index_config.sparse_index_dict(),
        "row_count": 0,
        "build_job": {
            "state": STATUS_BUILDING,
            "mode": mode or index_config.default_rebuild_mode,
            "total": total,
            "done": 0,
            "error": "",
            "started_at": _now(),
            "updated_at": _now(),
        },
        "validation": {},
        "eval_result_ref": "",
        "created_at": _now(),
        "activated_at": "",
        "retired_at": "",
    }
    try:
        coll.insert_one(doc)
        logger.info(f"已登记重建任务: {doc['physical']} (version={version}, mode={doc['mode']})")
        return as_dict(doc)
    except Exception as e:
        logger.error(f"登记重建任务失败: {e}")
        return None


def register_existing(logical: str = None, version: int = 1, physical: str = None,
                      row_count: int = 0, mode: str = "migrated",
                      dense_index: dict = None, sparse_index: dict = None,
                      embedding_model: str = "") -> Optional[dict]:
    """
    把「已经存在的物理集合」直接登记为 active——alias 迁移专用，不走 building 流程。
    迁移前线上集合就叫 logical 名，改造后它变成 {logical}_v1 并由 alias 指回来，
    注册表必须知道这件事，否则导入节点拿不到写入目标。
    :return: 记录；Mongo 不可用返回 None（迁移仍算成功，导入会退化成写 logical 名）
    """
    coll = _coll()
    logical = logical or index_config.logical_collection
    physical = physical or index_config.physical_name(version)

    existing = get(logical, version)
    if existing:
        logger.info(f"版本 {physical} 已在注册表中（状态 {existing.get('status')}），跳过登记")
        return existing

    if coll is None:
        logger.error("索引注册表不可用（Mongo 未连接），迁移后导入会退化写 alias 名")
        return None

    now = _now()
    doc = {
        "logical": logical,
        "alias": logical,
        "version": version,
        "physical": physical,
        "status": STATUS_ACTIVE,
        "mode": mode,
        "embedding_model": embedding_model,
        "dense_index": dense_index or index_config.dense_index_dict(),
        "sparse_index": sparse_index or index_config.sparse_index_dict(),
        "row_count": row_count,
        "build_job": {"state": STATUS_ACTIVE, "mode": mode, "total": row_count, "done": row_count,
                      "error": "", "started_at": now, "updated_at": now},
        "validation": {},
        "eval_result_ref": "",
        "created_at": now,
        "activated_at": now,
        "retired_at": "",
    }
    try:
        coll.insert_one(doc)
        logger.info(f"已登记现有集合为 active: {physical} (version={version}, rows={row_count})")
        return as_dict(doc)
    except Exception as e:
        logger.error(f"登记现有集合失败: {e}")
        return None


def set_status(logical: str, version: int, status: str, **extra) -> bool:
    """
    更新版本状态，可同时写入其它字段（row_count / eval_result_ref / validation 等）
    :return: 是否更新成功
    """
    coll = _coll()
    if coll is None:
        return False

    fields = {"status": status, "updated_at": _now(), "build_job.state": status}
    if status == STATUS_FAILED and extra.get("error"):
        fields["build_job.error"] = extra["error"]
    for key, value in extra.items():
        if key == "error":
            fields["build_job.error"] = value
        else:
            fields[key] = value

    try:
        result = coll.update_one({"logical": logical, "version": version}, {"$set": fields})
        if result.modified_count or result.matched_count:
            logger.info(f"{index_config.physical_name(version)} 状态 → {status}")
            return True
        logger.warning(f"状态更新未命中记录: logical={logical} version={version}")
        return False
    except Exception as e:
        logger.error(f"状态更新失败: {e}")
        return False


def update_progress(logical: str, version: int, done: int, total: int = None,
                    state: str = None, error: str = None) -> bool:
    """刷新重建进度（CLI 与 web 状态接口都读这里）"""
    coll = _coll()
    if coll is None:
        return False

    fields = {"build_job.done": done, "build_job.updated_at": _now()}
    if total is not None:
        fields["build_job.total"] = total
    if state:
        fields["build_job.state"] = state
    if error is not None:
        fields["build_job.error"] = error

    try:
        coll.update_one({"logical": logical, "version": version}, {"$set": fields})
        return True
    except Exception as e:
        logger.error(f"进度更新失败: {e}")
        return False


def activate(logical: str, version: int) -> bool:
    """
    把某个 ready 版本设为 active，并把原 active 置为 draining（保留回滚窗口）
    注意：本函数只改注册表，真正的 alias 切换由 rebuild.py 调 alter_alias 完成；
    顺序是「先 alter_alias 成功，再改注册表」，避免注册表说切了而线上没切。
    """
    coll = _coll()
    if coll is None:
        return False

    # 先确认目标版本存在：否则会把当前 active 降级成 draining，线上却没有任何 active
    target = get(logical, version)
    if not target:
        logger.error(f"激活失败：找不到 logical={logical} version={version}")
        return False

    try:
        previous = get_active(logical)
        now = _now()
        coll.update_many(
            {"logical": logical, "status": STATUS_ACTIVE},
            {"$set": {"status": STATUS_DRAINING, "updated_at": now}},
        )
        coll.update_one(
            {"logical": logical, "version": version},
            {"$set": {"status": STATUS_ACTIVE, "activated_at": now, "updated_at": now,
                      "build_job.state": STATUS_ACTIVE}},
        )
        if previous and previous.get("version") != version:
            logger.info(f"旧版本 {previous.get('physical')} → draining（{index_config.retire_after_days} 天后可清理）")
        logger.info(f"{target.get('physical')} → active")
        return True
    except Exception as e:
        logger.error(f"激活失败: {e}")
        return False


def mark_failed(logical: str, version: int, error: str) -> bool:
    """标记构建失败：退出双写，等待人工处理（新集合可以直接 drop）"""
    return set_status(logical, version, STATUS_FAILED, error=error)


def mark_physical_failed(physical: str, error: str) -> bool:
    """
    按物理集合名标记失败——双写副本写挂时导入节点只知道集合名，不知道版本号
    :return: 是否找到记录并标记成功（找不到说明它不是注册表管理的集合，忽略即可）
    """
    record = get_by_physical(physical)
    if not record:
        logger.warning(f"物理集合[{physical}]不在索引注册表中，跳过失败标记")
        return False
    return mark_failed(record["logical"], record["version"], error)


def retire(logical: str, version: int, dropped: bool = True) -> bool:
    """
    清理旧版本
    :param dropped: 物理集合是否已删除（False 表示只改状态，集合留给人工处理）
    """
    return set_status(logical, version, STATUS_RETIRED, retired_at=_now(),
                      **({} if dropped else {"note": "物理集合未删除，需人工确认"}))


def delete_version(logical: str, version: int) -> bool:
    """
    彻底删除一条注册表记录——只用于 alias 迁移回滚。
    回滚后物理集合又变回线上集合本体，记录留着会被清理任务当成「可删的旧版本」误删数据，
    所以这里必须是真删除而不是标成 retired。
    """
    coll = _coll()
    if coll is None:
        return False
    try:
        result = coll.delete_one({"logical": logical, "version": version})
        logger.info(f"已删除注册表记录: logical={logical} version={version}")
        return bool(result.deleted_count)
    except Exception as e:
        logger.error(f"删除注册表记录失败: {e}")
        return False


def list_retirable(logical: str = None) -> List[dict]:
    """列出已过观察期、可以清理的 draining 版本"""
    coll = _coll()
    if coll is None:
        return []
    logical = logical or index_config.logical_collection
    cutoff = (datetime.now() - timedelta(days=index_config.retire_after_days)).isoformat(timespec="seconds")
    return [
        as_dict(d) for d in coll.find({
            "logical": logical,
            "status": STATUS_DRAINING,
            "updated_at": {"$lte": cutoff},
        }).sort("version", ASCENDING)
    ]


# ==================== 给导入节点用 ====================

def get_write_targets(logical: str = None) -> List[str]:
    """
    当前应该写入哪些物理集合（导入节点的双写注入点）
    - 正常单写：[active 物理集合]
    - 重建期间双写：[active, building]
    - 注册表不可用/还没迁移：[.env 里的集合名]，等同改造前行为

    返回的是**物理集合名**而不是 alias：写入必须落到确定的物理集合，
    否则双写会两次都写进同一个 alias 指向的集合。
    """
    logical = logical or index_config.logical_collection

    active = get_active(logical)
    if not active:
        # 还没做过 alias 迁移，或者 Mongo 挂了 → 保持现状
        return [logical]

    targets = [active["physical"]]
    building = get_in_progress(logical)
    if building and index_config.double_write_enabled:
        # 只有 building / backfilling 阶段需要双写；validating/ready 阶段存量已齐，
        # 但仍继续双写直到切换完成，避免切换瞬间漏数据
        targets.append(building["physical"])

    # 去重保序
    return list(dict.fromkeys(targets))


def registry_status(logical: str = None) -> Dict[str, Any]:
    """给 /admin/index/status 用的汇总视图"""
    logical = logical or index_config.logical_collection
    versions = list_versions(logical)
    active = next((v for v in versions if v.get("status") == STATUS_ACTIVE), None)
    building = next((v for v in versions if v.get("status") in IN_PROGRESS), None)

    def brief(v: Optional[dict]) -> dict:
        if not v:
            return {}
        job = v.get("build_job") or {}
        total = job.get("total") or 0
        done = job.get("done") or 0
        return {
            "version": v.get("version"),
            "physical": v.get("physical"),
            "status": v.get("status"),
            "mode": v.get("mode"),
            "row_count": v.get("row_count"),
            "progress": round(done / total, 4) if total else None,
            "done": done,
            "total": total,
            "error": job.get("error") or "",
            "dense_index": v.get("dense_index"),
            "eval_result_ref": v.get("eval_result_ref"),
            "created_at": v.get("created_at"),
            "activated_at": v.get("activated_at"),
        }

    return {
        "logical": logical,
        "alias": logical,
        "double_write_enabled": index_config.double_write_enabled,
        "write_targets": get_write_targets(logical),
        "active": brief(active),
        "building": brief(building),
        "retire_after_days": index_config.retire_after_days,
        "retirable": [brief(v) for v in list_retirable(logical)],
        "history": [
            {"version": v.get("version"), "physical": v.get("physical"),
             "status": v.get("status"), "activated_at": v.get("activated_at")}
            for v in versions
        ],
        "registry_available": bool(versions) or _coll() is not None,
    }


if __name__ == "__main__":
    # 自测：需要 Mongo 在跑
    import json
    print(json.dumps(registry_status(), ensure_ascii=False, indent=2))
