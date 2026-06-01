"""
MCP Agent —— 支持通过 Model Context Protocol 调用外部工具的子 Agent。

向后兼容包装器：
  MCPAgent 现在继承统一的 Agent 基座，内部通过 MCPClientProvider 组合 MCP 能力。
  构造函数签名保持不变，所有现有调用方无需修改。

架构设计参考:
    - agenthub-py 的 mcp_agent_node: Supervisor 调度下，Agent 作为外部工具网关
    - hk-finance-mcp 的 FastMCP Server 模式: 工具注册、发现、执行标准化

核心能力:
    1. 在运行期通过 MCP 协议动态发现 Server 暴露的工具列表
    2. 将工具描述注入 Prompt，驱动 LLM 做 Tool Selection
    3. 解析 LLM 输出的 tool-call JSON，通过 MCP Client 执行工具
    4. 执行结果回传 LLM，生成最终自然语言回答（ReAct 风格）

连接方式（二选一）:
    - server_url : SSE 远程连接，例如 http://127.0.0.1:18080/sse
    - server_cmd : stdio 本地子进程，例如 ["python", "-m", "mcp_bridge.server.demo_server"]

示例:
    agent = MCPAgent(server_cmd=["python", "-m", "mcp_bridge.server.demo_server"])
    result = await agent.run("北京今天天气怎么样？")
"""
from typing import Any, Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from core.agents.base import Agent
from core.agents.toolkit import MCPClientProvider
from core.llm.model_type import ModelType
from core.utils.logger import get_logger

logger = get_logger(__name__)


class MCPAgent(Agent):
    """
    MCP-enabled Agent（向后兼容包装器）。

    继承 Agent 统一基座，内部组合 MCPClientProvider 实现 MCP 能力。
    构造函数和外部行为与旧版完全一致，调用方零改动。
    """

    name = "mcp"
    system_prompt = (
        "You are a helpful assistant that can use external tools to answer "
        "user questions. Analyze the request carefully and decide whether a tool "
        "is needed. If so, call the tool with correct arguments."
    )
    model_type = ModelType.GPT_4O_MINI

    def __init__(
        self,
        model_type: Optional[ModelType] = None,
        llm: Optional[BaseChatModel] = None,
        server_url: Optional[str] = None,
        server_cmd: Optional[List[str]] = None,
    ):
        if not server_url and not server_cmd:
            raise ValueError("MCPAgent 需要 server_url 或 server_cmd 之一来连接 MCP Server")

        self._provider = MCPClientProvider(server_url=server_url, server_cmd=server_cmd)
        super().__init__(model_type=model_type, llm=llm, tools=[self._provider])

        # 保留旧版属性以便外部访问（如需要调试）
        self.server_url = server_url
        self.server_cmd = server_cmd

    async def fetch_all(
        self,
        tool_name: str,
        arguments: dict,
        *,
        page_size: int = 50,
        max_pages: int = 20,
        max_records: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        自动分页拉取指定工具的全部数据。

        内部会独立建立 MCP 连接，执行翻页，完成后关闭连接。
        适合在 run() 之外直接获取全量数据做进一步分析。

        Args:
            tool_name: 工具名
            arguments: 基础参数（不要传 offset/limit）
            page_size: 每页条数
            max_pages: 最大翻页次数
            max_records: 最大累计条数

        Returns:
            合并后的 data 列表
        """
        async with self._provider:
            return await self._provider.fetch_all(
                tool_name=tool_name,
                arguments=arguments,
                page_size=page_size,
                max_pages=max_pages,
                max_records=max_records,
            )
