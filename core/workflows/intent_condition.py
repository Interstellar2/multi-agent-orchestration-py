"""
模式一：意图识别 + 条件路由

流程：Query -> IntentClassifier -> ConditionRouter -> Agent -> Output
"""
from typing import Dict, List, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel

from core.agents.base import Agent
from core.agents.resolver import resolve_agents
from core.llm.model_type import ModelType
from core.routing.condition import ConditionRouter
from core.routing.intent import IntentClassifier
from core.utils.logger import get_logger

logger = get_logger(__name__)


async def intent_condition_workflow(
    query: str,
    agents: Union[List[str], List[Agent]] = None,
    model_type: ModelType = None,
    llm: Optional[BaseChatModel] = None,
    classifier_model: ModelType = None,
    classifier_llm: Optional[BaseChatModel] = None,
) -> Dict:
    """
    意图识别 + 条件路由。

    参数:
        agents: 允许调用的 Agent，List[str] 或 List[Agent]
        model_type / llm: 当 agents 为 List[str] 时，用于创建 Agent
        classifier_model: 意图识别专用模型（默认 fallback 到 model_type）
        classifier_llm: 意图识别专用 LLM 实例（优先于 classifier_model）
    """
    logger.info(f"[Workflow] 启动意图识别+条件路由 | query={query[:80]}")
    agent_instances = resolve_agents(
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
