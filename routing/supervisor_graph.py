"""
Team Supervisor —— LangGraph 版本
把 Python 循环拆成独立的 Graph 节点，通过 Command(goto=...) 动态跳转。

Graph 结构：
    START -> coordinator -> (search_agent | code_agent | chat_agent | END)
    search_agent -> coordinator
    code_agent -> coordinator
    chat_agent -> coordinator
    ...

State 中维护了 called_agents / round_num / outputs，实现防循环和多轮协调。
"""
from typing import Annotated, Any, Dict, List, Literal, Optional

from langchain.schema import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from agents.base import Agent
from llm.factory import llm_factory
from llm.model_type import ModelType


# ---- Schema ----

class RouterOutput(BaseModel):
    next: str = Field(description="下一个要调用的 Agent 名称，或 END 结束")
    reason: str = Field(description="选择理由")


class SupervisorState(TypedDict):
    query: str
    outputs: Annotated[List[Dict[str, Any]], lambda l, r: l + r]
    called_agents: Annotated[List[str], lambda l, r: l + r]
    round_num: int
    final_output: str


# ---- Nodes ----

def create_coordinator_node(agents: List[Agent], llm: BaseChatModel, max_rounds: int = 3):
    """
    创建 Coordinator 节点。
    根据当前 State 决定下一步走向哪个 Agent，或 END。
    """
    agent_names = [a.name for a in agents]
    goto_type = Literal[tuple(agent_names + ["END"])]

    async def _node(state: SupervisorState) -> Command[goto_type]:
        # 防循环：超过最大轮数直接结束
        if state["round_num"] >= max_rounds:
            return Command(goto="END")

        available = [a for a in agents if a.name not in state["called_agents"]]
        if not available:
            return Command(goto="END")

        # 构建 prompt
        team_info = "\n".join(
            f"- {a.name}: {a.system_prompt[:80]}..." for a in available
        )
        called_info = (
            f"\nAlready called: {state['called_agents']}\n" if state["called_agents"] else ""
        )
        prompt = (
            "You are a team coordinator. Analyze the user's request and select "
            "the most appropriate agent to handle it.\n\n"
            f"Available agents:\n{team_info}\n"
            f"{called_info}\n"
            "You must output a JSON object with:\n"
            "- next: the agent name to call next, or 'END' if finished\n"
            "- reason: why you chose this agent"
        )

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=state["query"]),
        ]
        structured_llm = llm.with_structured_output(RouterOutput)
        decision = await structured_llm.ainvoke(messages)

        if decision.next == "END" or decision.next not in agent_names:
            return Command(goto="END")

        # 这里不执行 Agent，只决定走向；Agent 在各自节点中执行
        return Command(goto=decision.next)

    return _node


def create_agent_node(agent: Agent):
    """
    为每个子 Agent 创建一个 LangGraph 节点。
    执行 Agent.run() 后，更新 State 并回到 Coordinator。
    """
    async def _node(state: SupervisorState) -> Command[Literal["coordinator"]]:
        output = await agent.run(state["query"])
        return Command(
            update={
                "outputs": [
                    {
                        "round": state["round_num"] + 1,
                        "agent": agent.name,
                        "output": output,
                    }
                ],
                "called_agents": [agent.name],
                "round_num": state["round_num"] + 1,
            },
            goto="coordinator",
        )

    return _node


def create_end_node():
    """结束节点：汇总所有 Agent 输出为 final_output"""
    async def _node(state: SupervisorState) -> Dict[str, Any]:
        outputs = state["outputs"]
        final = outputs[-1]["output"] if outputs else "No agent was called."
        return {"final_output": final}

    return _node


# ---- Builder ----

def build_supervisor_graph(
    agents: List[Agent],
    model_type: ModelType = None,
    llm: Optional[BaseChatModel] = None,
    max_rounds: int = 3,
):
    """
    构建 LangGraph 版本的 Supervisor。

    用法：
        from llm.model_type import ModelType
        graph = build_supervisor_graph(
            agents=[SearchAgent(), CodeAgent()],
            model_type=ModelType.GPT_4O,
        )
        result = await graph.ainvoke({"query": "帮我写个 Python 函数"})
    """
    if llm is not None:
        llm_instance = llm
    else:
        llm_instance = llm_factory.get_model(model_type or ModelType.GPT_4O)

    builder = StateGraph(SupervisorState)

    # 注册 coordinator 节点
    builder.add_node("coordinator", create_coordinator_node(agents, llm_instance, max_rounds))

    # 注册每个子 Agent 节点
    for agent in agents:
        builder.add_node(agent.name, create_agent_node(agent))

    # 注册结束节点
    builder.add_node("end", create_end_node())

    # 边：START -> coordinator
    builder.add_edge(START, "coordinator")

    # coordinator 到各 Agent 和 END 的动态边（在节点内用 Command 控制，这里不需要显式 add_edge）
    # 但为了 LangGraph 能编译，需要把 coordinator 连接到所有可能的目标
    for agent in agents:
        builder.add_edge(agent.name, "coordinator")
    builder.add_edge("end", END)

    return builder.compile()


# ---- 便捷包装（保持和原版 TeamSupervisor 类似的调用接口） ----

class TeamSupervisorGraph:
    """
    和 TeamSupervisor 类似的调用接口，但底层是 LangGraph。
    支持可视化、断点续跑、人工介入等 LangGraph 原生能力。
    """

    def __init__(
        self,
        agents: List[Agent],
        model_type: ModelType = None,
        llm: Optional[BaseChatModel] = None,
        max_rounds: int = 3,
    ):
        self.agents = agents
        self.max_rounds = max_rounds
        self._graph = build_supervisor_graph(
            agents, model_type=model_type, llm=llm, max_rounds=max_rounds
        )

    async def run(self, query: str) -> Dict[str, Any]:
        state = {
            "query": query,
            "outputs": [],
            "called_agents": [],
            "round_num": 0,
            "final_output": "",
        }
        result = await self._graph.ainvoke(state)
        return {
            "final_output": result["final_output"],
            "history": result["outputs"],
            "called_agents": result["called_agents"],
            "mode": "team_supervisor_graph",
        }
