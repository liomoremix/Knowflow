# utils/task_utils.py

"""
任务状态管理工具模块(内存存储)

维护一个全局字典,记录每个任务/会话的:
- 全局状态 status: pending / processing / completed / failed
- 已完成节点列表 done_list、运行中节点列表 running_list(供前端轮询或 SSE progress 展示)
- 每会话结果 results(如非流式模式下暂存的 answer)

当 is_stream=True 时,状态/节点变更会自动向该会话推送一条 SSE progress 事件,
使聊天页进度面板实时更新(设计文档 doc 19 §2.3.1 / doc 20 §2.2.3)。
"""

import threading
from typing import Any, Dict, List, Optional

from utils.sse_utils import SSEEvent, push_to_session

TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

_lock = threading.Lock()
_tasks: Dict[str, Dict[str, Any]] = {}


def _ensure(key: str) -> Dict[str, Any]:
    if key not in _tasks:
        _tasks[key] = {
            "status": TASK_STATUS_PENDING,
            "done_list": [],
            "running_list": [],
            "results": {},
        }
    return _tasks[key]


def _push_progress(key: str, is_stream: bool) -> None:
    """is_stream 时把当前进度快照推给前端"""
    if not is_stream:
        return
    task = _tasks[key]
    push_to_session(
        key,
        SSEEvent.PROGRESS,
        {"done_list": list(task["done_list"]), "running_list": list(task["running_list"])},
    )


def update_task_status(key: str, status: str, is_stream: bool = False) -> None:
    with _lock:
        _ensure(key)["status"] = status
    _push_progress(key, is_stream)


def add_running_task(key: str, node_name: str, is_stream: bool = False) -> None:
    with _lock:
        task = _ensure(key)
        if node_name in task["done_list"]:
            task["done_list"].remove(node_name)
        if node_name not in task["running_list"]:
            task["running_list"].append(node_name)
    _push_progress(key, is_stream)


def add_done_task(key: str, node_name: str, is_stream: bool = False) -> None:
    with _lock:
        task = _ensure(key)
        if node_name in task["running_list"]:
            task["running_list"].remove(node_name)
        if node_name not in task["done_list"]:
            task["done_list"].append(node_name)
    _push_progress(key, is_stream)


def get_task_status(key: str) -> str:
    with _lock:
        return _ensure(key)["status"]


def get_done_task_list(key: str) -> List[str]:
    with _lock:
        return list(_ensure(key)["done_list"])


def get_running_task_list(key: str) -> List[str]:
    with _lock:
        return list(_ensure(key)["running_list"])


def set_task_result(key: str, field: str, value: Any) -> None:
    with _lock:
        _ensure(key)["results"][field] = value


def get_task_result(key: str, field: str, default: Any = None) -> Any:
    with _lock:
        return _ensure(key)["results"].get(field, default)
