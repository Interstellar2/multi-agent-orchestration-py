"""
香港法律 Agent 注册表

用法:
    from hk_law.agents import get_hk_law_agent, HK_LAW_AGENTS

    agent = get_hk_law_agent("criminal")
    output = await agent.run("我被控盗窃，该怎么办？")
"""
from typing import Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from hk_law.agents.base import HKLawAgent
from hk_law.agents.civil import CivilLawAgent
from hk_law.agents.company import CompanyLawAgent
from hk_law.agents.criminal import CriminalLawAgent
from hk_law.agents.employment import EmploymentLawAgent
from hk_law.agents.property import PropertyLawAgent
from core.llm.model_type import ModelType

# 所有可用法域 Agent 列表
HK_LAW_AGENTS: List[type[HKLawAgent]] = [
    CriminalLawAgent,
    CivilLawAgent,
    CompanyLawAgent,
    EmploymentLawAgent,
    PropertyLawAgent,
]

# 名称到类的映射
_AGENT_REGISTRY: Dict[str, type[HKLawAgent]] = {
    agent.name: agent for agent in HK_LAW_AGENTS
}


def get_hk_law_agent(
    name: str,
    model_type: ModelType = None,
    llm: Optional[BaseChatModel] = None,
    top_k: int = 5,
) -> HKLawAgent:
    """通过名称实例化香港法律 Agent"""
    agent_cls = _AGENT_REGISTRY.get(name)
    if not agent_cls:
        raise ValueError(
            f"Unknown HK law agent: {name}. "
            f"Available: {list(_AGENT_REGISTRY.keys())}"
        )
    return agent_cls(model_type=model_type, llm=llm, top_k=top_k)


def list_domains() -> List[str]:
    """列出所有可用法域"""
    return list(_AGENT_REGISTRY.keys())
