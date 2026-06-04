"""
Team Supervisor 模块（Python 原生循环版）
参考 agenthub-py team_agent_node，用 LLM 做 Supervisor，
根据任务动态选择 Agent，支持多轮调用。
"""
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from core.agents.base import Agent
from core.llm.factory import llm_factory
from core.llm.model_type import ModelType
from core.routing.supervisor_base import SupervisorPromptBuilder, AgentExecutionEngine
from core.utils.logger import get_logger

logger = get_logger(__name__)


class RouterOutput(BaseModel):
    next: str = Field(description="下一个要调用的 Agent 名称，或 END 结束")
    reason: str = Field(description="选择理由")


class TeamSupervisor:
    """
    Team Supervisor（Python 原生循环版）。
    管理一组 Agent，通过 LLM 动态决定调用哪个 Agent，支持多轮。

    如果需要用 LangGraph 可视化/持久化，请使用 supervisor_graph.py 中的版本。
    """

    def __init__(
        self,
        agents: List[Agent],
        model_type: ModelType = None,
        llm: Optional[BaseChatModel] = None,
        max_rounds: int = 3,
    ):
        self.agents = {a.name: a for a in agents}
        self.agent_list = agents
        self.max_rounds = max_rounds
        if llm is not None:
            self._llm = llm
        else:
            self._llm = llm_factory.get_model(model_type or ModelType.GPT_4O)

    async def run(self, query: str, history: Optional[list] = None) -> Dict[str, Any]:
        """
        运行 Supervisor 协调流程。

        支持串行链式调用：后调用的 Agent 能看到前面 Agent 的输出结果，
        从而基于已有信息继续处理，实现真正的多 Agent 协作。

        返回包含所有 Agent 输出和路由历史的字典。
        """
        logger.info(f"[Supervisor] 启动 | max_rounds={self.max_rounds} agents={list(self.agents.keys())}")
        results = []
        called = []
        round_num = 0
        accumulated_outputs = ""  # 累积前面所有 Agent 的输出，作为上下文传递

        while round_num < self.max_rounds:
            # 排除已调用过的 Agent（可选：允许重复调用则去掉这行）
            available = [a for a in self.agent_list if a.name not in called]

            if not available:
                logger.info("[Supervisor] 所有 Agent 已调用，结束")
                break

            # LLM 决策（复用公共 PromptBuilder）
            prompt = SupervisorPromptBuilder.build(
                query, available, called, accumulated_outputs
            )
            messages = [
                SystemMessage(content=prompt),
            ]
            structured_llm = self._llm.with_structured_output(RouterOutput)
            decision = await structured_llm.ainvoke(messages)
            logger.info(f"[Supervisor] 决策 | round={round_num + 1} next={decision.next} reason={decision.reason[:60]}")

            if decision.next == "END" or decision.next not in self.agents:
                logger.info("[Supervisor] 收到 END 或无效目标，结束")
                break

            if decision.next in called:
                logger.info(f"[Supervisor] Agent {decision.next} 已调用过，跳过")
                break

            # 执行选中的 Agent（复用公共 AgentExecutionEngine）
            agent = self.agents[decision.next]
            output = await AgentExecutionEngine.execute(
                agent, query, accumulated_outputs, history=history
            )

            results.append({
                "round": round_num + 1,
                "agent": agent.name,
                "reason": decision.reason,
                "output": output,
            })
            called.append(agent.name)

            # 累积输出到上下文，供下一个 Agent / Supervisor 使用
            accumulated_outputs += f"\n--- {agent.name} ---\n{output}\n"
            round_num += 1

        logger.info(f"[Supervisor] 完成 | 共调用 {len(called)} 个 agent: {called}")
        return {
            "final_output": results[-1]["output"] if results else "No agent was called.",
            "history": results,
            "called_agents": called,
            "context": accumulated_outputs,
        }
