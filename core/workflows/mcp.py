"""
模式四/五：MCP ReAct Agent 和 MCP Supervisor
"""
from typing import Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from core.agents.mcp_agent import MCPAgent
from core.agents.resolver import resolve_agents
from core.llm.model_type import ModelType
from core.routing.supervisor import TeamSupervisor
from core.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_MCP_CMD = ["python", "-m", "mcp_bridge.server.demo_server", "--transport", "stdio"]


async def mcp_react_workflow(
    query: str,
    server_url: Optional[str] = None,
    server_cmd: Optional[List[str]] = None,
    model_type: ModelType = None,
    llm: Optional[BaseChatModel] = None,
) -> Dict:
    """
    MCP ReAct Agent（外部工具调用演示）。

    流程:
        Query -> MCPAgent 连接 MCP Server -> 发现工具 -> LLM 决策 ->
        Tool Call (via MCP) -> 结果回传 -> 最终回答

    参数:
        server_url: SSE 远程地址，如 http://127.0.0.1:18080/sse
        server_cmd: stdio 本地命令，如 ["python", "-m", "mcp_bridge.server.demo_server"]
        model_type / llm: MCPAgent 使用的 LLM

    注意:
        server_url 和 server_cmd 二选一，优先 server_url。
    """
    logger.info(f"[Workflow] 启动 MCP ReAct | query={query[:80]}")

    if not server_url and not server_cmd:
        server_cmd = _DEFAULT_MCP_CMD.copy()
        logger.info(f"[Workflow] 未指定 MCP Server，使用默认 stdio 命令: {server_cmd}")

    mcp_agent = MCPAgent(
        model_type=model_type,
        llm=llm,
        server_url=server_url,
        server_cmd=server_cmd,
    )

    output = await mcp_agent.run(query)
    logger.info(f"[Workflow] 完成 MCP ReAct | output_len={len(output)}")
    return {
        "mode": "mcp_react",
        "agent": mcp_agent.name,
        "output": output,
    }


async def mcp_supervisor_workflow(
    query: str,
    server_url: Optional[str] = None,
    server_cmd: Optional[List[str]] = None,
    model_type: ModelType = None,
    llm: Optional[BaseChatModel] = None,
    supervisor_model: ModelType = None,
    supervisor_llm: Optional[BaseChatModel] = None,
    max_rounds: int = 3,
) -> Dict:
    """
    Team Supervisor + MCP Agent（多 Agent 协作 + 外部工具）。

    在标准 Supervisor 团队中注入 MCPAgent，使其可以调用外部 MCP Server。
    演示 Supervisor 如何将需要工具调用的任务路由给 MCPAgent。
    """
    logger.info(f"[Workflow] 启动 MCP Supervisor | query={query[:80]}")

    if not server_url and not server_cmd:
        server_cmd = _DEFAULT_MCP_CMD.copy()
        logger.info(f"[Workflow] 未指定 MCP Server，使用默认 stdio 命令: {server_cmd}")

    mcp_agent = MCPAgent(
        model_type=model_type,
        llm=llm,
        server_url=server_url,
        server_cmd=server_cmd,
    )
    base_agents = resolve_agents(
        ["search", "chat"],
        model_type=model_type,
        llm=llm,
    )
    all_agents = base_agents + [mcp_agent]

    supervisor = TeamSupervisor(
        agents=all_agents,
        model_type=supervisor_model or model_type,
        llm=supervisor_llm,
        max_rounds=max_rounds,
    )
    result = await supervisor.run(query)
    logger.info(f"[Workflow] 完成 MCP Supervisor | 调用 history={result.get('called_agents')}")
    return {"mode": "mcp_supervisor", **result}
