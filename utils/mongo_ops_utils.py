# utils/mongo_ops_utils.py

"""MongoDB 集合统一访问入口（运维类数据：索引注册表、查询日志、线上评测结果）。

与 utils/mongo_history_utils.py 的区别：
- 那个是对话历史的业务存储，import 期就建连（连不上直接抛异常，属于硬依赖）
- 这个是运维观测数据，**懒加载 + 永不抛异常**：Mongo 挂了也不能影响导入和查询主流程，
  最多是注册表/日志功能降级。

所有函数在连接失败时返回 None，调用方必须容错。
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

from tool.logger import logger

load_dotenv()

# serverSelectionTimeoutMS 设小一点：Mongo 不可用时不要卡住业务请求
CONNECT_TIMEOUT_MS = 3000

_client: Optional[MongoClient] = None
_db: Optional[Database] = None
_collections: Dict[str, Collection] = {}
_indexed: set = set()
# 连接失败后短路一段时间，避免每个请求都重试建连把日志刷爆
_failed = False


def get_mongo_db(reset: bool = False) -> Optional[Database]:
    """
    获取数据库对象（懒加载单例）
    :param reset: True 表示丢弃已有连接重新建连（连接被判定失效时用）
    :return: Database；连接失败返回 None
    """
    global _client, _db, _collections, _indexed, _failed
    if reset:
        _client = None
        _db = None
        _collections = {}
        _indexed = set()
        _failed = False

    if _db is not None:
        return _db
    if _failed:
        return None

    url = os.getenv("MONGO_URL")
    db_name = os.getenv("MONGO_DB_NAME")
    if not url or not db_name:
        logger.error("MONGO_URL / MONGO_DB_NAME 未配置，运维类 Mongo 功能不可用")
        _failed = True
        return None

    try:
        _client = MongoClient(url, serverSelectionTimeoutMS=CONNECT_TIMEOUT_MS)
        _db = _client[db_name]
        # 主动探活：不探的话首次失败会发生在真正的读写上，报错信息更难定位
        _client.admin.command("ping")
        logger.info(f"运维 Mongo 已连接: {db_name}")
        return _db
    except Exception as e:
        logger.error(f"运维 Mongo 连接失败（相关功能降级，不影响主流程）: {e}")
        _client = None
        _db = None
        _failed = True
        return None


def get_collection(name: str, indexes: Optional[List[Tuple[list, dict]]] = None) -> Optional[Collection]:
    """
    获取集合并确保索引存在（create_index 自带幂等性）
    :param name: 集合名
    :param indexes: [([("字段", 1/-1)], {索引选项}), ...]
    :return: Collection；连接失败返回 None
    """
    if name in _collections:
        return _collections[name]

    db = get_mongo_db()
    if db is None:
        return None

    try:
        collection = db[name]
        if indexes and name not in _indexed:
            for keys, options in indexes:
                collection.create_index(keys, **(options or {}))
            _indexed.add(name)
        _collections[name] = collection
        return collection
    except Exception as e:
        logger.error(f"获取 Mongo 集合 {name} 失败: {e}")
        return None


def mongo_available() -> bool:
    """运维 Mongo 是否可用（供健康检查/状态接口使用）"""
    return get_mongo_db() is not None


def as_dict(doc: Optional[dict]) -> dict:
    """Mongo 文档转可 JSON 序列化字典：去掉 _id 或转成字符串"""
    if not doc:
        return {}
    out = dict(doc)
    if "_id" in out:
        out["_id"] = str(out["_id"])
    return out
