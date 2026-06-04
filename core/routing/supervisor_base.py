"""
Team Supervisor 公共逻辑

提取 supervisor.py 和 supervisor_graph.py 共用的 prompt 构建、agent 执行逻辑，
消除两个 Supervisor 实现之间的重复代码。
"""
from typing import Any, Dict, List, Optional

from core.agents.base import Agent
from core.utils.logger import get_logger

logger = get_logger(__name__)


class SupervisorPromptBuilder:
    """
    Supervisor 决策 prompt 构建器。

    负责将 user query、available agents、已调用历史、前面 agent 的输出
    组装成给 LLM coordinator 的 system prompt。
    """

    @staticmethod
    def build(
        query: str,
        available_agents: List[Agent],
        called_agents: List[str],
        accumulated_outputs: str = "",
    ) -> str:
        team_info = "\n".join(
            f"- {a.name}: {a.system_prompt[:80]}..." for a in available_agents
        )
        called_info = f"\nAlready called: {called_agents}\n" if called_agents else ""
        context_info = (
            f"\nPrevious agent outputs:\n{accumulated_outputs}\n"
            if accumulated_outputs
            else ""
        )
        available_names = [a.name for a in available_agents]
        enum_hint = f"\nAllowed values for 'next': {available_names + ['END']}\n"

        return (
            "You are a team coordinator. Analyze the user's request and select "
            "the most appropriate agent to handle it.\n\n"
            f"User request: {query}\n"
            f"{context_info}"
            f"{called_info}\n"
            f"Available agents:\n{team_info}\n"
            f"{enum_hint}\n"
            "Rules:\n"
            "1. Do not call the same agent repeatedly.\n"
            "2. If the task is complete, set next to 'END'.\n"
            "3. You must output a JSON object with:\n"
            "   - next: the agent name to call next, or 'END' if finished\n"
            "   - reason: why you chose this agent"
        )


class AgentExecutionEngine:
    """
    Agent 执行引擎。

    负责构建增强 query（注入前面 agent 的输出上下文）并执行 agent.run()。
    """

    @staticmethod
    def build_enhanced_query(query: str, accumulated_outputs: str) -> str:
        """把前面 Agent 的输出作为上下文附加到原始 query。"""
        if accumulated_outputs:
            return (
                f"Original user request: {query}\n\n"
                f"Previous agent outputs:\n{accumulated_outputs}\n\n"
                f"Please continue to help with the original request."
            )
        return query

    @staticmethod
    def format_accumulated_outputs(outputs: List[Dict[str, Any]]) -> str:
        """将 Agent 输出列表格式化为上下文字符串。"""
        if not outputs:
            return ""
        lines = []
        for item in outputs:
            lines.append(f"--- {item['agent']} (round {item['round']}) ---")
            lines.append(item["output"])
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    async def execute(
        agent: Agent,
        query: str,
        accumulated_outputs: str = "",
        history: Optional[list] = None,
    ) -> str:
        """
        执行单个 Agent，自动附加历史上下文。

        Args:
            agent: 要执行的 Agent
            query: 原始用户请求
            accumulated_outputs: 前面所有 Agent 的格式化输出
            history: 多轮对话历史（用户对话层面）
        """
        enhanced_query = AgentExecutionEngine.build_enhanced_query(
            query, accumulated_outputs
        )
        logger.info(f"[AgentExecutionEngine] 执行 agent={agent.name}")
        return await agent.run(enhanced_query, history=history)
