"""
ToolProvider 接口与 MCP 实现

组合式工具能力：通过 ToolProvider 接口将外部工具调用能力（MCP、代码执行、API 调用等）
以插件形式注入 Agent，替代继承式耦合。

用法:
    provider = MCPClientProvider(server_url="http://127.0.0.1:1888/mcp/sse")
    async with provider:
        tools = await provider.discover()
        result = await provider.invoke("query_by_stock_code", {"code": "00700"})
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client

from core.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ToolInfo:
    """工具元信息"""
    name: str
    description: str
    parameters: Dict[str, Any]


class ToolProvider(ABC):
    """
    工具提供者抽象基类。

    子类需要实现：
      - discover(): 发现可用工具列表
      - invoke(): 执行指定工具
      - __aenter__ / __aexit__: 管理连接生命周期（可选，但推荐）
    """

    @abstractmethod
    async def discover(self) -> List[ToolInfo]:
        """发现可用工具列表"""
        ...

    @abstractmethod
    async def invoke(self, tool_name: str, arguments: dict) -> str:
        """调用指定工具，返回文本结果"""
        ...

    async def __aenter__(self):
        """异步上下文入口（可选）"""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步上下文出口（可选）"""
        return False


class MCPClientProvider(ToolProvider):
    """
    MCP 协议工具提供者。

    封装 MCP Client 的连接、工具发现、调用链路，
    作为可插拔组件注入 Agent 基座。

    连接方式（二选一）:
      - server_url: SSE 远程连接，例如 http://127.0.0.1:18080/sse
      - server_cmd: stdio 本地子进程，例如 ["python", "-m", "mcp_bridge.server.demo_server"]
    """

    def __init__(
        self,
        server_url: Optional[str] = None,
        server_cmd: Optional[List[str]] = None,
    ):
        if not server_url and not server_cmd:
            raise ValueError("MCPClientProvider 需要 server_url 或 server_cmd 之一")

        self.server_url = server_url
        self.server_cmd = server_cmd
        self._session: Optional[ClientSession] = None

    # ------------------------------------------------------------------
    # 生命周期管理
    # ------------------------------------------------------------------

    async def __aenter__(self):
        if self.server_url:
            logger.info(f"[MCPClientProvider] 建立 SSE 连接 | {self.server_url}")
            self._streams_ctx = sse_client(self.server_url)
            streams = await self._streams_ctx.__aenter__()
            # 兼容 mcp 1.x sse_client 返回格式
            if isinstance(streams, tuple) and len(streams) == 2:
                read_stream, write_stream = streams
            else:
                read_stream, write_stream = streams, streams
        else:
            params = StdioServerParameters(
                command=self.server_cmd[0],
                args=self.server_cmd[1:] if len(self.server_cmd) > 1 else [],
                env=None,
            )
            logger.info(f"[MCPClientProvider] 建立 stdio 连接 | {self.server_cmd}")
            self._streams_ctx = stdio_client(params)
            read_stream, write_stream = await self._streams_ctx.__aenter__()

        self._session_ctx = ClientSession(read_stream, write_stream)
        self._session = await self._session_ctx.__aenter__()
        await self._session.initialize()
        logger.info("[MCPClientProvider] MCP Session 初始化完成")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session_ctx is not None:
            await self._session_ctx.__aexit__(exc_type, exc_val, exc_tb)
        if self._streams_ctx is not None:
            await self._streams_ctx.__aexit__(exc_type, exc_val, exc_tb)
        self._session = None
        logger.info("[MCPClientProvider] MCP Session 已关闭")
        return False

    # ------------------------------------------------------------------
    # ToolProvider 接口实现
    # ------------------------------------------------------------------

    async def discover(self) -> List[ToolInfo]:
        if self._session is None:
            raise RuntimeError("MCPClientProvider 未进入异步上下文，请先 async with")

        tools_response = await self._session.list_tools()
        result = []
        for tool in tools_response.tools:
            schema = getattr(tool, "inputSchema", {})
            result.append(
                ToolInfo(
                    name=getattr(tool, "name", "unknown"),
                    description=getattr(tool, "description", "No description."),
                    parameters=schema.get("properties", {}),
                )
            )
        logger.info(f"[MCPClientProvider] 发现 {len(result)} 个工具")
        return result

    async def invoke(self, tool_name: str, arguments: dict) -> str:
        if self._session is None:
            raise RuntimeError("MCPClientProvider 未进入异步上下文，请先 async with")

        logger.info(f"[MCPClientProvider] 调用工具 | {tool_name}({arguments})")
        try:
            tool_result = await self._session.call_tool(tool_name, arguments=arguments)
            result_text = "\n".join(
                item.text for item in tool_result.content if hasattr(item, "text")
            )
            return result_text
        except Exception as exc:
            logger.error(f"[MCPClientProvider] 工具调用失败 | {exc}")
            raise
