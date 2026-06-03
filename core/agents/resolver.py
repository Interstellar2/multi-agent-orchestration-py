"""
Agent 解析器

将字符串名称或 Agent 实例列表统一解析为 Agent 实例列表。
原本在 workflows.py 中作为私有函数，现提取为公共模块供多处复用。
"""
from typing import List, Optional, Union

from langchain_core.language_models.chat_models import BaseChatModel

from core.agents.base import Agent
from core.agents.specialized import get_agent
from core.llm.model_type import ModelType


def resolve_agents(
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
