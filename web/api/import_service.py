import os
import shutil
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel, Field

from config.index_config import index_config
from config.minio_config import minio_config
from processor.import_processor.main_graph import KBImportWorkflow
from tool.logger import logger
from utils.index_registry_utils import registry_status
from utils.metrics_utils import observe_import_task, register_ops_collector
from utils.minio_utils import get_minio_client
from utils.task_utils import (
    add_done_task,
    add_running_task,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
    update_task_status,
)

load_dotenv()

# 1. 创建应用
# 标题和描述会在Swagger文档中展示
app = FastAPI(
    title="知识库问答系统-导入API",
    description="此文档是知识库问答系统导入流程的API接口说明"
)

# 1.5 Prometheus 指标：/metrics 供 deploy/monitoring 里的 Prometheus 抓取
register_ops_collector()
Instrumentator().instrument(app).expose(app, include_in_schema=False)

# 2. 跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许的源
    allow_credentials=True,  # 允许携带cookie
    allow_methods=["*"],  # 允许的请求方法
    allow_headers=["*"],  # 允许的请求头
)

# 3. 静态页面路由：返回文件导入前端页面
# 访问地址：http://localhost:8000/import.html
@app.get("/import.html")  # 对外访问地址
async def get_import_page():
    # 拼接HTML文件绝对路径
    current_dir_parent_path = Path(__file__).absolute().parent.parent
    html_path = current_dir_parent_path / "page" / "import.html"
    # 如果不存在，抛出404异常
    if not html_path.exists():
        raise HTTPException(status_code=404, detail=f"没有查询到页面，地址为：{html_path}")
    return FileResponse(html_path)

# 4. 后台任务：LangGraph全流程执行
# 独立于主请求线程，由BackgroundTasks触发，避免阻塞接口响应
def run_graph_task(task_id: str, file_dir: str, import_file_path: str):
    """
    LangGraph全流程执行后台任务
    核心流程：初始化状态 → 流式执行图节点 → 实时更新任务状态 → 异常捕获
    任务状态更新：pending → processing → completed/failed
    节点进度更新：每完成一个节点，将节点名加入done_list，供前端轮询查看

    :param task_id: 全局唯一任务ID，关联单个文件的全流程处理
    :param file_dir: 该任务的本地文件存储目录（含临时文件/解析结果）
    :param import_file_path: 上传文件的本地绝对路径
    """
    try:
        # 1. 更新任务全局状态为：处理中
        update_task_status(task_id, "processing")

        # 2. 初始化LangGraph状态
        init_state = {
            "task_id": task_id,
            "file_dir": file_dir,
            "import_file_path": import_file_path,
        }

        # 3. 流式执行LangGraph全流程（stream模式：实时获取每个节点的执行结果）
        workflow = KBImportWorkflow()
        for event in  workflow.run(init_state, stream=True):
            for node_name, node_result in event.items():
                # 将完成的节点名加入【已完成列表】，前端轮询/status/{task_id}可实时获取
                add_done_task(task_id, node_name)

        # 4. 全流程执行完成，更新任务全局状态为：已完成
        update_task_status(task_id, "completed")
        observe_import_task("completed")

    except Exception as e:
        # 5. 捕获全流程异常，更新任务全局状态为：失败，并记录错误日志（含堆栈）
        update_task_status(task_id, "failed")
        observe_import_task("failed")
        logger.info(f"[{task_id}] LangGraph全流程执行失败，异常信息：{str(e)}", exc_info=True)

# 5. 核心接口：文件上传接口
# 支持多文件上传，核心流程：接收文件 → 本地保存 → MinIO上传 → 启动后台任务
# 访问地址：http://localhost:8000/upload （POST请求，form-data格式传参）
@app.post("/upload", summary="文件上传接口", description="支持多文件批量上传，自动触发知识库导入全流程")
async def upload_files(background_tasks: BackgroundTasks, files: List[UploadFile] = File(...)):
    """
    文件上传核心接口
    1. 接收前端上传的多文件（PDF/MD为主）
    2. 按「日期/任务ID」分层保存到本地输出目录，避免文件冲突
    3. 将文件上传至MinIO对象存储，做持久化保存
    4. 为每个文件生成唯一TaskID，启动独立的LangGraph后台处理任务
    5. 实时更新任务状态，供前端轮询监控进度

    :param background_tasks: FastAPI后台任务对象，用于异步执行LangGraph流程
    :param files: 前端上传的文件列表（form-data格式）
    :return: 包含上传结果和所有任务ID的JSON响应
    """
    # 1. 构建本地存储根目录：项目根目录/doc/YYYYMMDD（按日期分层，方便管理）
    data_based_root_dir = os.getenv("DATA_BASED_ROOT_DIR")
    data_dir = os.path.join(data_based_root_dir, datetime.now().strftime("%Y%m%d"))
    # 初始化任务ID列表，用于返回给前端（一个文件对应一个TaskID）
    task_ids = []

    # 2. 遍历处理每个上传的文件（多文件批量处理，各自独立生成TaskID）
    for file in files:
        # 生成全局唯一TaskID（UUID4），作为单个文件的全流程标识
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        logger.info(f"[{task_id}] 开始处理上传文件，文件名：{file.filename}，文件类型：{file.content_type}")

        # 3. 标记「文件上传」阶段为「运行中」，前端轮询可查
        add_running_task(task_id, "upload_file")

        # 4. 构建该任务的本地独立目录：output/YYYYMMDD/TaskID，避免多文件重名冲突
        file_dir = os.path.join(data_dir, task_id)
        os.makedirs(file_dir, exist_ok=True)  # 目录不存在则创建，存在则不做处理
        # 构建上传文件的本地保存绝对路径
        import_file_path = os.path.join(file_dir, file.filename)

        # 5. 将上传的文件保存到本地临时目录（后续MinIO上传/文件解析均基于此文件）
        with open(import_file_path, "wb") as file_buffer:
            shutil.copyfileobj(file.file, file_buffer)
        logger.info(f"[{task_id}] 文件已保存至本地，路径：{import_file_path}")

        # 6. 将本地文件上传至MinIO对象存储，做持久化保存
        # 构建MinIO中的文件对象名：pdf_files/YYYYMMDD/文件名（按日期分层，和本地一致）
        minio_object_name = f"pdf_files/{datetime.now().strftime('%Y%m%d')}/{file.filename}"
        try:
            # 获取MinIO客户端实例
            minio_client = get_minio_client()

            # 从环境变量获取MinIO的桶名配置
            minio_bucket_name = minio_config.bucket_name

            # 本地文件上传至MinIO（同名文件会自动覆盖，保证文件最新）
            minio_client.fput_object(
                bucket_name=minio_bucket_name,
                object_name=minio_object_name,
                file_path=import_file_path,
                content_type=file.content_type  # 传递文件原始MIME类型
            )
            logger.info(f"[{task_id}] 文件已成功上传至MinIO，桶名：{minio_bucket_name}，对象名：{minio_object_name}")
        except Exception as e:
            # MinIO上传失败，记录警告日志（不中断后续流程，本地文件仍可继续处理）
            logger.warning(f"[{task_id}] 文件上传MinIO失败，将继续执行本地处理流程，异常信息：{str(e)}", exc_info=True)

        # 7. 标记「文件上传」阶段为「已完成」，前端轮询可查
        add_done_task(task_id, "upload_file")

        # 8. 将LangGraph全流程处理加入FastAPI后台任务（异步执行，不阻塞当前接口响应）
        background_tasks.add_task(run_graph_task, task_id, file_dir, import_file_path)
        logger.info(f"[{task_id}] 已将LangGraph全流程加入后台任务，任务已启动")

    # 9. 所有文件处理完毕，返回上传成功信息和所有TaskID（前端基于TaskID轮询进度）
    logger.info(f"多文件上传处理完毕，共处理{len(files)}个文件，生成TaskID列表：{task_ids}")
    return {
        "code": 200,
        "message": f" 文件上传成功, total: {len(files)}",
        "task_ids": task_ids
    }

# 6. 核心接口：任务状态查询接口
# 前端轮询此接口获取单个任务的处理进度和状态
# 访问地址：http://localhost:8000/status/{task_id} （GET请求）
@app.get("/status/{task_id}", summary="任务状态查询", description="根据TaskID查询单个文件的处理进度和全局状态")
async def get_task_progress(task_id: str):
    """
    任务状态查询接口
    前端轮询此接口（如每秒1次），获取任务的实时处理进度
    返回数据均来自内存中的任务管理字典（task_utils.py），高性能无IO

    :param task_id: 全局唯一任务ID（由/upload接口返回）
    :return: 包含任务全局状态、已完成节点、运行中节点的JSON响应
    """
    # 构造任务状态返回体
    task_status_info: Dict[str, Any] = {
        "code": 200,
        "task_id": task_id,
        "status": get_task_status(task_id),  # 任务全局状态：pending/processing/completed/failed
        "done_list": get_done_task_list(task_id),  # 已完成的节点/阶段列表
        "running_list": get_running_task_list(task_id)  # 正在运行的节点/阶段列表
    }
    # 记录状态查询日志，方便追踪前端轮询情况
    logger.info(
        f"[{task_id}] 任务状态查询，当前状态：{task_status_info['status']}，已完成节点：{task_status_info['done_list']}")
    return task_status_info


# 7. 健康检查接口：前端据此点亮「API 已连接」状态，并与查询服务保持一致
# 访问地址：http://localhost:8000/health （GET请求）
@app.get("/health")
async def health():
    """证明服务已正常启动"""
    return {"ok": True}


# ==================== 索引管理接口（蓝绿双索引） ====================
#
# 为什么进度不在内存里：重建是分钟级任务，可能由 CLI 发起、也可能由本接口发起，
# 而 task_utils 是进程内字典，跨进程看不见。所以状态一律从 Mongo 的 index_registry 读，
# 谁来问都是同一份答案。

# 同一时刻只允许一个重建任务，避免两个请求各建一个版本互相双写
_rebuild_state = {"running": False, "started_at": "", "finished_at": "", "exit_code": None, "argv": []}
_rebuild_lock = threading.Lock()


class RebuildRequest(BaseModel):
    """重建请求参数，字段与 rebuild.py 的 CLI 一一对应"""
    mode: str = Field("move_vectors", description="move_vectors=搬旧向量（快），re_embed=重算向量（换模型时用）")
    prepare_only: bool = Field(False, description="True 表示建好验好停在 ready，不自动切别名")
    eval_limit: int = Field(20, description="门禁评测每场景取样条数")
    skip_eval: bool = Field(False, description="跳过召回门禁（不建议）")
    M: Optional[int] = Field(None, description="覆盖 HNSW 的 M")
    ef_construction: Optional[int] = Field(None, description="覆盖 HNSW 的 efConstruction")
    drop_on_failure: bool = Field(False, description="失败时自动删除新集合")


class SwitchRequest(BaseModel):
    """切换/回滚请求参数"""
    version: int = Field(..., description="要切换到的版本号")


class SchedulerRequest(BaseModel):
    """巡检请求参数，字段与 scheduler.py 的 CLI 一一对应"""
    sample: Optional[int] = Field(None, description="线上抽样条数，0 表示跳过线上评测")
    auto_rebuild: Optional[bool] = Field(None, description="判定需要重建时是否自动执行，默认读 .env")
    force: bool = Field(False, description="已有在建版本时仍然触发重建")
    no_online: bool = Field(False, description="跳过线上抽样评测，只看索引侧信号")


def _run_rebuild(argv: List[str]):
    """后台执行重建，只负责记录本进程的执行结果（真正的进度在 Mongo 里）"""
    from processor.index_manager import rebuild

    try:
        code = rebuild.main(argv)
    except Exception as e:
        logger.error(f"索引重建后台任务异常: {e}", exc_info=True)
        code = 1
    finally:
        with _rebuild_lock:
            _rebuild_state.update({"running": False, "exit_code": code,
                                   "finished_at": datetime.now().isoformat(timespec="seconds")})
        logger.info(f"索引重建后台任务结束，退出码 {code}")


@app.get("/admin/index/status", summary="索引状态", description="别名指向、各版本状态、重建进度、可清理版本")
async def admin_index_status(quality: bool = False):
    """
    查索引注册表状态
    :param quality: True 时附带索引质量代理指标（要连 Milvus ORM 读 segment，稍慢）
    """
    payload: Dict[str, Any] = {"code": 200, **registry_status()}
    payload["task"] = dict(_rebuild_state)
    if quality:
        from processor.index_manager.quality import index_quality
        payload["quality"] = index_quality()
    return payload


@app.post("/admin/index/rebuild", summary="发起蓝绿重建",
          description="后台执行：建新集合→双写→搬存量→一致性校验→召回门禁→切换别名")
async def admin_index_rebuild(background_tasks: BackgroundTasks, request: RebuildRequest):
    """发起一次重建，立刻返回；进度用 GET /admin/index/status 轮询"""
    with _rebuild_lock:
        if _rebuild_state["running"]:
            raise HTTPException(status_code=409, detail="已有重建任务在执行，请等它结束")
        if _scheduler_state["running"]:
            # 巡检跑到一半时启重建：这一轮采到的全是过渡状态（双写中、行数对不上），
            # 判定结论没有意义，而且巡检自己也可能正打算触发一个重建
            raise HTTPException(status_code=409, detail="索引巡检进行中，等它结束再重建")
        _rebuild_state.update({"running": True, "exit_code": None, "finished_at": "",
                               "started_at": datetime.now().isoformat(timespec="seconds")})

    argv = ["--yes", "--mode", request.mode, "--eval-limit", str(request.eval_limit)]
    if request.prepare_only:
        argv.append("--prepare-only")
    if request.skip_eval:
        argv.append("--skip-eval")
    if request.drop_on_failure:
        argv.append("--drop-on-failure")
    if request.M:
        argv += ["--M", str(request.M)]
    if request.ef_construction:
        argv += ["--ef-construction", str(request.ef_construction)]

    with _rebuild_lock:
        _rebuild_state["argv"] = argv
    logger.info(f"收到索引重建请求: {argv}")
    background_tasks.add_task(_run_rebuild, argv)

    return {"code": 202, "message": "重建已在后台启动，用 GET /admin/index/status 查看进度", "argv": argv}


@app.post("/admin/index/switch", summary="切换别名到指定版本",
          description="人工确认后把线上流量切到某个 ready/draining 版本，回滚也是这个接口")
async def admin_index_switch(request: SwitchRequest):
    """切换别名。prepare_only 建好的版本靠它上线；出问题切回旧版本也是它"""
    from processor.index_manager import rebuild
    from utils.milvus_utils import get_milvus_client

    client = get_milvus_client()
    if not client:
        raise HTTPException(status_code=503, detail="Milvus 连接失败")

    code = rebuild.do_switch(client, index_config.logical_collection, request.version)
    if code != 0:
        raise HTTPException(status_code=400, detail=f"切换失败（退出码 {code}），详情见服务日志")
    return {"code": 200, "message": f"已切换到 version={request.version}", **registry_status()}


# ==================== 巡检调度接口（P5） ====================
#
# 常驻巡检应该单独跑：python -m processor.index_manager.scheduler --loop
# 这里的接口是给「手动跑一轮」用的（演示、排查、验证判定逻辑），
# 一轮要调几十次 LLM 当裁判，是分钟级任务，所以同样走后台执行 + Mongo 存结果。

_scheduler_state = {"running": False, "started_at": "", "finished_at": "", "decision": None}
_scheduler_lock = threading.Lock()


def _run_scheduler(sample: Optional[int], auto_rebuild: Optional[bool], force: bool, no_online: bool):
    """后台跑一轮巡检，只记录本进程的执行状态（历史结论一律从 Mongo 读）"""
    from processor.index_manager import scheduler

    try:
        cycle = scheduler.run_cycle(sample=sample, auto_rebuild=auto_rebuild, force=force,
                                    with_online=not no_online)
        decision = cycle.get("decision")
    except Exception as e:
        logger.error(f"巡检后台任务异常: {e}", exc_info=True)
        decision = f"error: {e}"
    finally:
        with _scheduler_lock:
            _scheduler_state.update({"running": False, "decision": decision,
                                     "finished_at": datetime.now().isoformat(timespec="seconds")})


@app.get("/admin/scheduler/status", summary="巡检状态", description="本轮执行状态 + 最近几轮的判定结论")
async def admin_scheduler_status(limit: int = 3):
    """查巡检历史。结论从 Mongo 读，所以 CLI 跑的轮次这里也看得见"""
    from processor.index_manager.scheduler import recent_cycles

    return {"code": 200, "task": dict(_scheduler_state), "cycles": recent_cycles(limit)}


@app.post("/admin/scheduler/run", summary="手动跑一轮巡检",
          description="后台执行：线上抽样评测→采集索引/延迟信号→判定→（可选）触发带门禁的重建")
async def admin_scheduler_run(background_tasks: BackgroundTasks, request: SchedulerRequest):
    """立刻返回；结论用 GET /admin/scheduler/status 轮询"""
    with _scheduler_lock:
        if _scheduler_state["running"]:
            raise HTTPException(status_code=409, detail="已有巡检在执行，请等它结束")
        if _rebuild_state["running"]:
            # 重建期间采到的信号全是过渡状态（双写、行数不一致），判定没有意义
            raise HTTPException(status_code=409, detail="索引重建进行中，等它结束再巡检")
        _scheduler_state.update({"running": True, "decision": None, "finished_at": "",
                                 "started_at": datetime.now().isoformat(timespec="seconds")})

    logger.info(f"收到巡检请求: sample={request.sample} auto_rebuild={request.auto_rebuild} "
                f"force={request.force} no_online={request.no_online}")
    background_tasks.add_task(_run_scheduler, request.sample, request.auto_rebuild,
                              request.force, request.no_online)
    return {"code": 202, "message": "巡检已在后台启动，用 GET /admin/scheduler/status 查看结论"}


if __name__ == "__main__":
    uvicorn.run(app=app,host="127.0.0.1",port=8000)