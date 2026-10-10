# utils/sse_utils.py

"""
SSE(Server-Sent Events)推送工具模块

为每个会话(session_id)维护一个 asyncio.Queue,作为后端向浏览器单向推送事件的管道。
- 图节点运行在 FastAPI BackgroundTasks 的工作线程中,而队列消费发生在事件循环里,
  因此 push_to_session 必须线程安全(通过 loop.call_soon_threadsafe 投递)。
- sse_generator 是异步生成器,由 StreamingResponse 驱动,按 SSE 协议输出
  "event: <name>\\ndata: <json>\\n\\n" 帧。
"""

import asyncio
import json
from enum import Enum
from typing import Any, Dict, Optional

from tool.logger import logger


class SSEEvent(Enum):
    """SSE 事件类型,枚举值即前端 addEventListener 监听的事件名"""
    READY = "ready"        # 连接握手成功
    PROGRESS = "progress"  # 节点进度(done_list / running_list)
    DELTA = "delta"        # 流式增量文本
    FINAL = "final"        # 完整最终答案(含 image_urls)
    ERROR = "error"        # 错误


# session_id -> {"queue": asyncio.Queue, "loop": 事件循环或 None, "closed": bool}
_sessions: Dict[str, Dict[str, Any]] = {}


def create_sse_queue(session_id: str) -> None:
    """
    为会话创建结果队列。应在事件循环内(如 /query 接口)调用,
    以便记录当前 loop,供工作线程线程安全地投递消息。
    """
    loop: Optional[asyncio.AbstractEventLoop] = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    _sessions[session_id] = {"queue": asyncio.Queue(), "loop": loop, "closed": False}
    logger.debug(f"[SSE] 为会话 {session_id} 创建队列")


def push_to_session(session_id: str, event: SSEEvent, data: Dict[str, Any]) -> None:
    """
    向指定会话推送一个事件。线程安全:可在工作线程中调用。
    会话不存在或已关闭时静默丢弃(不抛异常,避免影响主链路)。
    """
    sess = _sessions.get(session_id)
    if sess is None or sess["closed"]:
        return
    msg = {"event": event.value, "data": data}
    loop = sess.get("loop")
    queue = sess["queue"]
    if loop is not None and loop.is_running():
        loop.call_soon_threadsafe(queue.put_nowait, msg)
    else:
        queue.put_nowait(msg)


def _frame(event: SSEEvent, data: Dict[str, Any]) -> str:
    """拼装一帧 SSE 报文"""
    return f"event: {event.value}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def sse_generator(session_id: str, request) -> Any:
    """
    SSE 异步生成器:先发送 ready 握手帧,随后持续从会话队列取消息并输出,
    收到 final / error 或客户端断开时结束并清理会话。
    """
    sess = _sessions.get(session_id)
    if sess is None:
        create_sse_queue(session_id)
        sess = _sessions[session_id]
    queue = sess["queue"]

    yield _frame(SSEEvent.READY, {"session_id": session_id})
    try:
        while True:
            if await request.is_disconnected():
                logger.debug(f"[SSE] 会话 {session_id} 客户端已断开")
                break
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=30)
            except asyncio.TimeoutError:
                # 长时间无消息时发送注释行保活,防止代理/浏览器切断连接
                yield ": keep-alive\n\n"
                continue
            yield _frame(SSEEvent(msg["event"]), msg["data"])
            if msg["event"] in (SSEEvent.FINAL.value, SSEEvent.ERROR.value):
                break
    finally:
        sess["closed"] = True
        _sessions.pop(session_id, None)
        logger.debug(f"[SSE] 会话 {session_id} 推送结束,已清理")
