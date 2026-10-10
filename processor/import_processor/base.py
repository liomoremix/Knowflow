# processor/import_processor/base.py

"""
导入流程节点基类

定义统一的节点接口规范，提供通用功能
"""
import colorlog

from processor.import_processor.config import ImportConfig, get_config
from processor.import_processor.exceptions import ImportProcessError
from utils.metrics_utils import node_timer
from utils.task_utils import add_done_task, add_running_task

"""
导入流程节点基类

定义统一的节点接口规范，提供通用功能
"""

from abc import ABC, abstractmethod
from typing import TypeVar, Optional
import logging

T = TypeVar("T")  # 泛型状态类型

"""
1、抽象的类（ABC），子类需要实现这个类
2、抽象的方法（@abstractmethod process），子类需要实现这个方法
3、name属性，子类必须覆盖这个属性
4、__init__() 方法中初始化了日志记录器，日志记录器的名字是 self所在类的名字
5、__init__() 方法中初始化了全局配置，获取全局单例配置对象
6、__call__() 方法的调用时机  
    (1)对象 = 类名() 
    (2)结果 = 对象()     圆括号调用了__call__()方法
7、self.process()， 当前这句话是从哪调用过来的，process()就定义在哪个位置
8、未来可以在父类的process()中实现任务追踪
9、封装了统一日志处理，使用logging、colorlog
    注意：使用日志记录器记录日志时一定要先激活日志setup_logging(logging.INFO)，并指定日志级别

"""


class BaseNode(ABC):
    """
    导入流程节点基类

    所有节点类都应继承此基类，实现 process 方法。
    基类提供统一的日志、任务追踪和错误处理。

    使用示例:
        class MyNode(BaseNode):
            name = "my_node"

            def process(self, state):
                # 实现具体逻辑
                return state

        # 作为 LangGraph 节点使用
        node = MyNode()
        state = node()
        workflow.add_node("my_node", node)
    """

    name: str = "base_node"  # 节点名称，子类应覆盖

    def __init__(self, config: Optional[ImportConfig] = None):
        """
        初始化节点

        Args:
            config: 配置对象，默认使用全局配置
        """
        self.config = config or get_config()

        # 日志记录器：命名 import.{self.name}
        self.logger = logging.getLogger(f"import.{self.name}")

    def __call__(self, state: T) -> T:
        """
        节点执行入口

        LangGraph 调用节点时会调用此方法。
        提供统一的日志输出、任务追踪和异常处理。

        Args:
            state: 图状态字典

        Returns:
            更新后的状态字典

        Raises:
            ImportProcessError: 节点执行失败时抛出
        """
        # 指标埋点：与查询侧同一个入口约定，导入链路的节点耗时/错误率由此统一采集
        with node_timer("import", self.name):
            try:
                # 1. 开始准备执行节点
                self.logger.info(f"--- {self.name} 开始啦 ---")

                # 任务追踪:记录节点进入运行中(供前端轮询/SSE 展示进度)
                task_id = state.get("task_id")
                if task_id:
                    add_running_task(task_id, self.name)

                # 2. 执行节点
                result = self.process(state)

                # 任务追踪:记录节点完成
                if task_id:
                    add_done_task(task_id, self.name)

                # 3. 执行节点成功
                self.logger.info(f"--- {self.name} 完成啦 ---")

                return result

            except Exception as e:
                self.logger.error(f"{self.name} 执行失败: {e}")
                raise ImportProcessError(
                    message="节点失败",
                    node_name=self.name,
                    cause=e
                )

    @abstractmethod
    def process(self, state: T) -> T:
        """
        节点核心处理逻辑

        子类必须实现此方法。

        Args:
            state: 图状态字典

        Returns:
            更新后的状态字典
        """
        pass

    def log_step(self, step_name: str, message: str = ""):
        """
        记录步骤日志
            [步骤名称]传递进来的消息
        Args:
            step_name: 步骤名称
            message: 附加信息
        """
        log_msg = f"[{step_name}]"
        if message:
            log_msg += f" {message}"
        self.logger.info(log_msg)


# 配置日志格式
def setup_logging(level: int = logging.INFO):
    """
    配置日志格式

    Args:
        level: 日志级别
    """

    logger = logging.getLogger()
    logger.setLevel(level)

    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',  # INFO 显示为绿色
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'bold_red',
        }
    ))

    logger.handlers.clear()
    logger.addHandler(handler)
