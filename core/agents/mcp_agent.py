"""
MCP Agent —— 支持通过 Model Context Protocol 调用外部工具的子 Agent。

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
import json
import re
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

from core.agents.base import Agent
from core.llm.factory import llm_factory
from core.llm.model_type import ModelType
from core.utils.logger import get_logger

logger = get_logger(__name__)


class MCPAgent(Agent):
    """
    MCP-enabled Agent。

    继承 Agent 基类的 LLM 初始化逻辑，额外封装 MCP Client 的
    连接、工具发现、调用链路。
    """

    name = "mcp"
    system_prompt = (
        "You are a helpful assistant that can use external tools to answer "
        "user questions. Analyze the request carefully and decide whether a tool "
        "is needed. If so, call the tool with correct arguments."
    )
    model_type = ModelType.GPT_4O_MINI

    # ReAct 循环最大工具调用轮数
    MAX_TOOL_ROUNDS = 3

    def __init__(
        self,
        model_type: Optional[ModelType] = None,
        llm: Optional[BaseChatModel] = None,
        server_url: Optional[str] = None,
        server_cmd: Optional[List[str]] = None,
    ):
        super().__init__(model_type=model_type, llm=llm)

        if not server_url and not server_cmd:
            raise ValueError("MCPAgent 需要 server_url 或 server_cmd 之一来连接 MCP Server")

        self.server_url = server_url
        self.server_cmd = server_cmd

    # -----------------------------------------------------------------------
    # 公共接口
    # -----------------------------------------------------------------------

    async def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        执行 MCP-enabled ReAct loop。

        流程:
            1. 连接 MCP Server（SSE / stdio）
            2. 发现工具列表 (list_tools)
            3. 构造带工具描述的 system prompt
            4. 最多 MAX_TOOL_ROUNDS 轮: LLM -> 解析 tool call -> 执行 -> 结果回传
            5. 返回最终自然语言回答
        """
        logger.info(f"[{self.name}] 启动 MCP ReAct | query={query[:80]}")

        if self.server_url:
            async with sse_client(self.server_url) as streams:
                # mcp 1.x sse_client 返回的是 (read_stream, write_stream) 或 Session
                # 兼容处理：统一按元组解包
                if isinstance(streams, tuple) and len(streams) == 2:
                    read_stream, write_stream = streams
                else:
                    read_stream, write_stream = streams, streams
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    return await self._react_loop(session, query, context)
        else:
            params = StdioServerParameters(
                command=self.server_cmd[0],
                args=self.server_cmd[1:] if len(self.server_cmd) > 1 else [],
                env=None,
            )
            async with stdio_client(params) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    return await self._react_loop(session, query, context)

    # -----------------------------------------------------------------------
    # 内部实现
    # -----------------------------------------------------------------------

    async def _react_loop(
        self,
        session: ClientSession,
        query: str,
        context: Optional[Dict[str, Any]],
    ) -> str:
        """ReAct 主循环。"""
        tools = await session.list_tools()
        tools_desc = self._format_tools(tools.tools)

        system_text = (
            f"{self.system_prompt}\n\n"
            f"You have access to the following tools:\n{tools_desc}\n\n"
            "When you need to use a tool, output ONLY a JSON object in this exact format:\n"
            '{"tool": "tool_name", "arguments": {"arg1": "value1"}}\n\n'
            "After receiving the tool result, answer the user in natural language. "
            "If no tool is needed, answer directly."
        )

        messages = [SystemMessage(content=system_text), HumanMessage(content=query)]

        for round_num in range(self.MAX_TOOL_ROUNDS):
            logger.info(f"[{self.name}] ReAct round {round_num + 1}/{self.MAX_TOOL_ROUNDS}")
            response = await self._llm.ainvoke(messages)
            content = str(response.content) if response.content else ""

            tool_call = self._extract_tool_call(content)
            if tool_call is None:
                # LLM 直接给出了最终回答
                logger.info(f"[{self.name}] ReAct 完成（无需更多工具）| rounds={round_num + 1}")
                return content

            # 执行工具调用
            tool_name = tool_call.get("tool")
            tool_args = tool_call.get("arguments", {})
            logger.info(f"[{self.name}] 调用工具 | {tool_name}({tool_args})")

            try:
                tool_result = await session.call_tool(tool_name, arguments=tool_args)
                # tool_result 是 CallToolResult，content 是 list[TextContent | ImageContent]
                result_text = "\n".join(
                    item.text for item in tool_result.content if hasattr(item, "text")
                )
            except Exception as exc:
                result_text = f"Tool execution error: {exc}"
                logger.error(f"[{self.name}] 工具调用失败 | {exc}")

            # 将工具调用与结果追加到对话历史
            messages.append(HumanMessage(content=content))  # LLM 的 tool-call 意图
            messages.append(
                SystemMessage(
                    content=f"[Tool Result: {tool_name}]\n{result_text}\n"
                    "Now provide the final answer to the user."
                )
            )

        # 达到最大轮数，强制总结
        logger.warning(f"[{self.name}] ReAct 达到最大轮数，强制总结")
        final_response = await self._llm.ainvoke(messages)
        return str(final_response.content) if final_response.content else ""

    # -----------------------------------------------------------------------
    # 工具描述格式化
    # -----------------------------------------------------------------------

    @staticmethod
    def _format_tools(tools: List[Any]) -> str:
        """将 MCP Tool 列表格式化为人类可读的文本描述。"""
        lines = []
        for tool in tools:
            name = getattr(tool, "name", "unknown")
            desc = getattr(tool, "description", "No description.")
            schema = getattr(tool, "inputSchema", {})
            # 简化 schema 展示：只列 properties 的 key
            props = schema.get("properties", {})
            args_desc = ", ".join(props.keys()) if props else "none"
            lines.append(f"- {name}({args_desc}): {desc}")
        return "\n".join(lines)

    # -----------------------------------------------------------------------
    # Tool-Call 提取
    # -----------------------------------------------------------------------

    @staticmethod
    def _extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
        """
        从 LLM 输出中提取 tool-call JSON。

        支持格式:
            1. 纯 JSON 对象（整段文本）
            2. Markdown code block 内的 JSON
        """
        text = text.strip()
        if not text:
            return None

        # 尝试整段解析
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "tool" in data and "arguments" in data:
                return data
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown code block 中提取
        pattern = re.compile(r"```(?:json)?\s*({[\s\S]*?})\s*```", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and "tool" in data and "arguments" in data:
                    return data
            except json.JSONDecodeError:
                pass

        # 尝试从文本中找最外层 JSON 对象（宽松匹配）
        brace_match = re.search(r"({[\s\S]*?\"tool\"[\s\S]*?})", text)
        if brace_match:
            try:
                data = json.loads(brace_match.group(1))
                if isinstance(data, dict) and "tool" in data and "arguments" in data:
                    return data
            except json.JSONDecodeError:
                pass

        return None
