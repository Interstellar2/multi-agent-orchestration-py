"""
模式二/三：Team Supervisor（Python 循环版 / LangGraph 版）
"""
from typing import Dict, List, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel

from core.agents.base import Agent
from core.agents.resolver import resolve_agents
from core.llm.model_type import ModelType
from core.routing.supervisor import TeamSupervisor
from core.routing.supervisor_graph import TeamSupervisorGraph
from core.utils.logger import get_logger

logger = get_logger(__name__)


async def team_supervisor_workflow(
    query: str,
    agents: Union[List[str], List[Agent]] = None,
    model_type: ModelType = None,
    llm: Optional[BaseChatModel] = None,
    supervisor_model: ModelType = None,
    supervisor_llm: Optional[BaseChatModel] = None,
    max_rounds: int = 3,
    history: Optional[list] = None,
    event_callback=None,
) -> Dict:
    """
    Team Supervisor（Python 原生循环版）。

    参数:
        supervisor_model: Coordinator 决策专用模型（默认 fallback 到 model_type）
        supervisor_llm: Coordinator 专用 LLM 实例（优先于 supervisor_model）
        history: 多轮对话历史（BaseMessage 列表）
        event_callback: 事件回调（用于 SSE 推送）
    """
    logger.info(f"[Workflow] 启动 Team Supervisor | query={query[:80]}")
    agent_instances = resolve_agents(
        agents or ["search", "code", "chat", "analysis"],
        model_type=model_type,
        llm=llm,
    )
    supervisor = TeamSupervisor(
        agents=agent_instances,
        model_type=supervisor_model or model_type,
        llm=supervisor_llm,
        max_rounds=max_rounds,
        event_callback=event_callback,
    )
    result = await supervisor.run(query, history=history)
    logger.info(f"[Workflow] 完成 Team Supervisor | 调用 history={result.get('called_agents')}")
    return {"mode": "team_supervisor", **result}


async def team_supervisor_graph_workflow(
    query: str,
    agents: Union[List[str], List[Agent]] = None,
    model_type: ModelType = None,
    llm: Optional[BaseChatModel] = None,
    supervisor_model: ModelType = None,
    supervisor_llm: Optional[BaseChatModel] = None,
    max_rounds: int = 3,
    history: Optional[list] = None,
    event_callback=None,
) -> Dict:
    """
    Team Supervisor（LangGraph 版本）。
    底层是 LangGraph StateGraph，支持可视化、持久化、断点续跑。

    参数:
        history: 多轮对话历史（BaseMessage 列表）
        event_callback: 事件回调（用于 SSE 推送）
    """
    logger.info(f"[Workflow] 启动 Team Supervisor (LangGraph) | query={query[:80]}")
    agent_instances = resolve_agents(
        agents or ["search", "code", "chat", "analysis"],
        model_type=model_type,
        llm=llm,
    )
    supervisor = TeamSupervisorGraph(
        agents=agent_instances,
        model_type=supervisor_model or model_type,
        llm=supervisor_llm,
        max_rounds=max_rounds,
        event_callback=event_callback,
    )
    result = await supervisor.run(query, history=history)
    logger.info(f"[Workflow] 完成 Team Supervisor (LangGraph) | 调用 history={result.get('called_agents')}")
    return result
