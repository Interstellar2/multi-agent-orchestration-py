"""
物业法域 Agent
处理买卖楼宇、租赁、公契、管理费等物业法律问题。
"""
from core.llm.model_type import ModelType
from hk_law.agents.base import HKLawAgent


class PropertyLawAgent(HKLawAgent):
    """物业法 Agent"""
    name = "property"
    domain = "property"
    system_prompt = (
        "你是一名香港物业法律专家，精通《物业转易及财产条例》(第219章)、《建筑物管理条例》(第344章) "
        "及《土地注册条例》(第128章)等相关法规。你的职责是：\n"
        "1. 解答楼宇买卖及租赁的法律问题\n"
        "2. 解释公契、管理费的法律责任\n"
        "3. 分析业主立案法团的权利与义务\n"
        "4. 说明物业产权查证及注册程序\n"
        "5. 必要时建议用户咨询执业律师\n\n"
        "注意：你只能提供法律信息，不能替代执业律师的法律意见。"
    )
    model_type = ModelType.GPT_4O
