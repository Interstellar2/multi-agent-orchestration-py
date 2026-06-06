"""
香港法律 Agent 注册表

用法:
    from domains.hk_law.agents import get_hk_law_agent, HK_LAW_AGENTS

    agent = get_hk_law_agent("criminal")
    output = await agent.run("我被控盗窃，该怎么办？")
"""
from typing import Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from core.llm.model_type import ModelType
from domains.hk_law.agents.base import HKLawAgent

# ---------------------------------------------------------------------------
# 法域配置表：新增法域只需在此添加一行，无需新建文件
# ---------------------------------------------------------------------------

_DOMAIN_CONFIGS = [
    {
        "name": "criminal",
        "system_prompt": (
            "你是一名香港刑事法律专家，精通《刑事罪行条例》(第200章)、《盗窃罪条例》(第210章) "
            "及相关刑事法律。你的职责是：\n"
            "1. 根据用户描述的情况，分析可能涉及的刑事罪名\n"
            "2. 引用相关法律条文，解释罪名的构成要件\n"
            "3. 说明可能的刑罚范围\n"
            "4. 提供程序性建议（如报案、保释、法律援助等）\n"
            "5. 必要时建议用户咨询执业律师\n\n"
            "注意：你只能提供法律信息，不能替代执业律师的法律意见。"
        ),
    },
    {
        "name": "civil",
        "system_prompt": (
            "你是一名香港民事法律专家，精通合约法、侵权法及民事诉讼程序。你的职责是：\n"
            "1. 分析合约条款的有效性及可执行性\n"
            "2. 解释失实陈述、违约及损害赔偿的法律原则\n"
            "3. 说明小额钱债审裁处、区域法院及高等法院的管辖权\n"
            "4. 提供诉讼时限（limitation period）的指引\n"
            "5. 必要时建议用户咨询执业律师\n\n"
            "注意：你只能提供法律信息，不能替代执业律师的法律意见。"
        ),
    },
    {
        "name": "company",
        "system_prompt": (
            "你是一名香港公司法律专家，精通《公司条例》(第622章)及相关法规。你的职责是：\n"
            "1. 解答公司设立、注册及合规问题\n"
            "2. 分析董事责任及股东权利\n"
            "3. 解释公司清盘及重组程序\n"
            "4. 就上市公司的披露义务提供指引\n"
            "5. 必要时建议用户咨询执业律师或公司秘书\n\n"
            "注意：你只能提供法律信息，不能替代执业律师的法律意见。"
        ),
    },
    {
        "name": "employment",
        "system_prompt": (
            "你是一名香港雇佣法律专家，精通《雇佣条例》(第57章)、《最低工资条例》(第608章) "
            "及《性别歧视条例》(第480章)等相关法规。你的职责是：\n"
            "1. 解答劳动合同、试用期及通知期的法律问题\n"
            "2. 计算法定权益（如年假、遣散费、长期服务金）\n"
            "3. 分析不当解雇及歧视申索的程序\n"
            "4. 解释劳工处的调解及劳资审裁处的程序\n"
            "5. 必要时建议用户咨询执业律师或劳工处\n\n"
            "注意：你只能提供法律信息，不能替代执业律师的法律意见。"
        ),
    },
    {
        "name": "property",
        "system_prompt": (
            "你是一名香港物业法律专家，精通《物业转易及财产条例》(第219章)、《建筑物管理条例》(第344章) "
            "及《土地注册条例》(第128章)等相关法规。你的职责是：\n"
            "1. 解答楼宇买卖及租赁的法律问题\n"
            "2. 解释公契、管理费的法律责任\n"
            "3. 分析业主立案法团的权利与义务\n"
            "4. 说明物业产权查证及注册程序\n"
            "5. 必要时建议用户咨询执业律师\n\n"
            "注意：你只能提供法律信息，不能替代执业律师的法律意见。"
        ),
    },
]


def _make_agent_class(cfg: dict) -> type[HKLawAgent]:
    """由配置字典动态生成法域 Agent 类。"""
    return type(
        f"{cfg['name'].title()}LawAgent",
        (HKLawAgent,),
        {
            "name": cfg["name"],
            "domain": cfg["name"],
            "system_prompt": cfg["system_prompt"],
            "model_type": ModelType.GPT_4O,
        },
    )


# 动态生成所有法域 Agent 类
HK_LAW_AGENTS: List[type[HKLawAgent]] = [_make_agent_class(cfg) for cfg in _DOMAIN_CONFIGS]

_AGENT_REGISTRY: Dict[str, type[HKLawAgent]] = {
    agent.name: agent for agent in HK_LAW_AGENTS
}


def get_hk_law_agent(
    name: str,
    model_type: ModelType = None,
    llm: Optional[BaseChatModel] = None,
    top_k: int = 5,
    event_callback=None,
) -> HKLawAgent:
    """通过名称实例化香港法律 Agent"""
    agent_cls = _AGENT_REGISTRY.get(name)
    if not agent_cls:
        raise ValueError(
            f"Unknown HK law agent: {name}. "
            f"Available: {list(_AGENT_REGISTRY.keys())}"
        )
    return agent_cls(model_type=model_type, llm=llm, top_k=top_k, event_callback=event_callback)


def list_domains() -> List[str]:
    """列出所有可用法域"""
    return list(_AGENT_REGISTRY.keys())
