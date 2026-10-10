# processor/query_processor/base.py

"""
查询流程节点基类

定义统一的节点接口规范，提供通用功能
"""
from abc import abstractmethod, ABC
from typing import TypeVar
from tool.logger import logger
from utils.metrics_utils import node_timer
from utils.task_utils import add_running_task

T = TypeVar("T")  # 泛型状态类型
class NodeBase(ABC):

    name: str = "base_node"  # 节点名称，子类应覆盖

    def __call__(self, state: T) -> T:
        """
        节点执行入口
        """
        # 指标埋点：所有节点共用这一个入口，加在这里就等价于「每个节点都有耗时和错误率」
        # 成功失败都会记耗时（失败往往是慢的一种），异常另外单独计数
        with node_timer("query", self.name):
            try:
                # 1. 开始准备执行节点
                logger.info(f"--- {self.name} 开始啦 ---")

                # 任务追踪:记录节点进入运行中(驱动 SSE progress / 前端进度面板)
                session_id = state.get("session_id")
                if session_id:
                    add_running_task(session_id, self.name, state.get("is_stream"))

                # 2. 执行节点
                result = self.process(state)

                # 3. 执行节点成功
                logger.info(f"--- {self.name} 完成啦 ---")

                return result

            except Exception as e:
                logger.error(f"{self.name} 执行失败: {e}")
                raise

    @abstractmethod
    def process(self, state: T) -> T:
        """
        节点核心处理逻辑
        子类必须实现此方法
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        pass