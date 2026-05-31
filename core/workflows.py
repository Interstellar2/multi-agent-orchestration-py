"""
工作流组合模块
将意图识别、路由、Agent 调用组合成完整的工作流。

核心设计：通过 ModelType 枚举 + 工厂统一管理 LLM，
每个组件（Classifier / Supervisor / Agent）可独立指定模型类型。

用法:
    from llm.model_type import ModelType
    from agents.specialized import CodeAgent, ChatAgent

    # 每个 Agent 配不同模型
    code_agent = CodeAgent(model_type=ModelType.QWEN_MAX)
    chat_agent = ChatAgent(model_type=ModelType.GPT_4O_MINI)

    # Supervisor 用最强的模型做决策
    result = await team_supervisor_graph_workflow(
        query,
        agents=[code_agent, chat_agent],
        supervisor_model=ModelType.GPT_4O,
    )
"""
from typing import Dict, List, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel

from core.agents.base import Agent
from core.agents.specialized import get_agent
from core.llm.model_type import ModelType
from core.routing.condition import ConditionRouter
from core.routing.intent import IntentClassifier
from core.routing.supervisor import TeamSupervisor
from core.routing.supervisor_graph import TeamSupervisorGraph
from core.utils.logger import get_logger

logger = get_logger(__name__)


def _resolve_agents(
    agents: Union[List[str], List[Agent], None],
    model_type: ModelType = None,
    llm: Optional[BaseChatModel] = None,
) -> List[Agent]:
    """
    统一解析 agents 参数。
    List[str]   -> 用 get_agent 逐个实例化，共享 model_type/llm
    List[Agent] -> 已经是实例，直接返回（每个可自带不同模型）
    """
    if agents is None or len(agents) == 0:
        return []
    if isinstance(agents[0], str):
        return [get_agent(name, model_type=model_type, llm=llm) for name in agents]  # type: ignore
    return agents  # type: ignore


async def intent_condition_workflow(
    query: str,
    agents: Union[List[str], List[Agent]] = None,
    model_type: ModelType = None,
    llm: Optional[BaseChatModel] = None,
    classifier_model: ModelType = None,
    classifier_llm: Optional[BaseChatModel] = None,
) -> Dict:
    """
    模式一：意图识别 + 条件路由
    流程：Query -> IntentClassifier -> ConditionRouter -> Agent -> Output

    参数:
        agents: 允许调用的 Agent，List[str] 或 List[Agent]
        model_type / llm: 当 agents 为 List[str] 时，用于创建 Agent
        classifier_model: 意图识别专用模型（默认 fallback 到 model_type）
        classifier_llm: 意图识别专用 LLM 实例（优先于 classifier_model）
    """
    logger.info(f"[Workflow] 启动意图识别+条件路由 | query={query[:80]}")
    agent_instances = _resolve_agents(
        agents or ["search", "code", "chat"], model_type=model_type, llm=llm
    )
    agent_names = [a.name for a in agent_instances]

    # 1. 意图识别（可独立指定模型）
    classifier = IntentClassifier(
        intents=agent_names,
        intent_descriptions={
            "search": "用户想要搜索信息、查询资料",
            "code": "用户想要写代码、调试程序",
            "chat": "用户想要闲聊或一般问答",
            "analysis": "用户想要数据分析",
        },
        model_type=classifier_model or model_type,
        llm=classifier_llm,
    )
    intent_result = await classifier.classify(query)

    # 2. 条件路由
    router = ConditionRouter.from_intent_map(
        {name: name for name in agent_names},
        default=agent_names[-1] if agent_names else None,
    )
    target = router.route(intent_result.intent)
    logger.info(f"[Workflow] 路由结果 | intent={intent_result.intent} -> target={target}")

    # 3. 执行对应 Agent（从实例列表中找到目标）
    target_agent = next((a for a in agent_instances if a.name == target), None)
    if target_agent is None:
        raise ValueError(f"No agent found for target: {target}")
    output = await target_agent.run(query)

    logger.info(f"[Workflow] 完成意图识别+条件路由 | target={target}")
    return {
        "mode": "intent_condition",
        "intent": intent_result.intent,
        "confidence": intent_result.confidence,
        "reason": intent_result.reason,
        "routed_to": target,
        "output": output,
    }


async def team_supervisor_workflow(
    query: str,
    agents: Union[List[str], List[Agent]] = None,
    model_type: ModelType = None,
    llm: Optional[BaseChatModel] = None,
    supervisor_model: ModelType = None,
    supervisor_llm: Optional[BaseChatModel] = None,
    max_rounds: int = 3,
) -> Dict:
    """
    模式二：Team Supervisor（Python 原生循环版）

    参数:
        supervisor_model: Coordinator 决策专用模型（默认 fallback 到 model_type）
        supervisor_llm: Coordinator 专用 LLM 实例（优先于 supervisor_model）
    """
    logger.info(f"[Workflow] 启动 Team Supervisor | query={query[:80]}")
    agent_instances = _resolve_agents(
        agents or ["search", "code", "chat", "analysis"],
        model_type=model_type,
        llm=llm,
    )
    supervisor = TeamSupervisor(
        agents=agent_instances,
        model_type=supervisor_model or model_type,
        llm=supervisor_llm,
        max_rounds=max_rounds,
    )
    result = await supervisor.run(query)
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
) -> Dict:
    """
    模式三：Team Supervisor（LangGraph 版本）
    底层是 LangGraph StateGraph，支持可视化、持久化、断点续跑。
    """
    logger.info(f"[Workflow] 启动 Team Supervisor (LangGraph) | query={query[:80]}")
    agent_instances = _resolve_agents(
        agents or ["search", "code", "chat", "analysis"],
        model_type=model_type,
        llm=llm,
    )
    supervisor = TeamSupervisorGraph(
        agents=agent_instances,
        model_type=supervisor_model or model_type,
        llm=supervisor_llm,
        max_rounds=max_rounds,
    )
    result = await supervisor.run(query)
    logger.info(f"[Workflow] 完成 Team Supervisor (LangGraph) | 调用 history={result.get('called_agents')}")
    return result
