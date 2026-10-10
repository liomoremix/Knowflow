# processor/query_processor/nodes/node_web_search_mcp.py

import asyncio
import json
import time

from agents.mcp import MCPServerStreamableHttp

from config.bailian_mcp_config import mcp_config
from processor.query_processor.base import NodeBase
from processor.query_processor.state import QueryGraphState
from tool.logger import logger
from utils.json_format_utils import serialize_json
from utils.metrics_utils import observe_node_error, observe_retrieval
from utils.query_trace_utils import path_metric


class NodeWebSearchMcp(NodeBase):
    """
    节点功能，调用外部搜索引擎补充信息
    """

    # 覆盖基类的 name 属性，标识节点名称
    name: str = "node_web_search_mcp"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        节点逻辑
        :param state: 工作流状态对象
        :return: 更新后的状态对象
        """
        # 意图策略开关：关闭联网搜索时直接短路
        strategy = state.get("strategy") or {}
        if not strategy.get("enable_web_search", True):
            logger.info("意图策略关闭联网搜索，跳过")
            return {"web_search_docs": [], "web_path_metric": path_metric("web", skipped=True)}

        query = state.get("rewritten_query", "")
        if not query:
            # 没有查询内容 = 这一路压根没跑，按 skipped 记，不进指标
            # （记 0 命中 + 0 耗时会同时把召回和 p95 带偏）
            logger.info("无查询内容，跳过联网搜索")
            return {"web_path_metric": path_metric("web", skipped=True)}

        # 计时覆盖 MCP 连接 + 调用：这一路的耗时几乎全在外部服务上，是最容易拖慢整体的一路
        started = time.perf_counter()
        docs = []
        error = ""
        try:
            result = asyncio.run(self._mcp_call(query))
            if result:
                pages = json.loads(result.content[0].text).get("pages") or []
                # 统一输出结构化结果，供后续 rerank/引用使用
                # 每条：{title, url, snippet}

                for item in pages:
                    snippet = (item.get("snippet") or "").strip()
                    url = (item.get("url") or "").strip()
                    title = (item.get("title") or "").strip()
                    if not snippet:
                        continue
                    docs.append({"title": title, "url": url, "snippet": snippet})

                logger.info("MCP 搜索结果:", docs)
        except Exception as e:
            # 联网搜索失败不该让整条查询失败（本地知识库还在），但必须留下可观测的痕迹
            error = str(e)[:200]
            logger.error(f"MCP 联网搜索失败，跳过该路: {e}")

        elapsed_ms = (time.perf_counter() - started) * 1000
        if error:
            # 这一路自己吞了异常，基类的错误计数不会触发，所以在这里补一次；
            # 失败不进检索指标，理由同上
            observe_node_error("query", self.name)
        else:
            observe_retrieval("web", elapsed_ms / 1000, len(docs))
        metric = path_metric("web", hits=len(docs), latency_ms=elapsed_ms,
                             params={"error": error} if error else {})
        # 保持原有行为：没有结果时不写 web_search_docs，下游按缺省处理
        if docs:
            return {"web_search_docs": docs, "web_path_metric": metric}
        return {"web_path_metric": metric}

    async def _mcp_call(self, query):

        search_mcp = MCPServerStreamableHttp(
            name="search_mcp",
            params={
                "url": mcp_config.mcp_base_url,
                "headers": {"Authorization": f"Bearer {mcp_config.api_key}"},
                "timeout": 10,
            },
            cache_tools_list=True,
            max_retry_attempts=3,
        )

        try:
            await search_mcp.connect()
            result = await search_mcp.call_tool(
                tool_name="bailian_web_search",
                arguments={"query": query, "count": 5},
            )
            return result
        finally:
            await search_mcp.cleanup()






if __name__ == "__main__":

    init_state = {
        "rewritten_query": "关于brother HAK180烫金机，如何调节转印温度？"
    }

    # 执行节点的业务调用
    node_web_search_mcp = NodeWebSearchMcp()
    result = node_web_search_mcp(init_state)
    logger.info(serialize_json(result, indent=4))
