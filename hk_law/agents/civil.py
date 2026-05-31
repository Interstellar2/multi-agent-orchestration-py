"""
民事/合约法域 Agent
处理合约纠纷、侵权、债务追讨等民事法律问题。
"""
from core.llm.model_type import ModelType
from hk_law.agents.base import HKLawAgent


class CivilLawAgent(HKLawAgent):
    """民事/合约法 Agent"""
    name = "civil"
    domain = "civil"
    system_prompt = (
        "你是一名香港民事法律专家，精通合约法、侵权法及民事诉讼程序。你的职责是：\n"
        "1. 分析合约条款的有效性及可执行性\n"
        "2. 解释失实陈述、违约及损害赔偿的法律原则\n"
        "3. 说明小额钱债审裁处、区域法院及高等法院的管辖权\n"
        "4. 提供诉讼时限（limitation period）的指引\n"
        "5. 必要时建议用户咨询执业律师\n\n"
        "注意：你只能提供法律信息，不能替代执业律师的法律意见。"
    )
    model_type = ModelType.GPT_4O
