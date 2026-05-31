"""
刑事法域 Agent
处理刑事案件、刑事程序、量刑等法律问题。
"""
from core.llm.model_type import ModelType
from hk_law.agents.base import HKLawAgent


class CriminalLawAgent(HKLawAgent):
    """刑事法 Agent"""
    name = "criminal"
    domain = "criminal"
    system_prompt = (
        "你是一名香港刑事法律专家，精通《刑事罪行条例》(第200章)、《盗窃罪条例》(第210章) "
        "及相关刑事法律。你的职责是：\n"
        "1. 根据用户描述的情况，分析可能涉及的刑事罪名\n"
        "2. 引用相关法律条文，解释罪名的构成要件\n"
        "3. 说明可能的刑罚范围\n"
        "4. 提供程序性建议（如报案、保释、法律援助等）\n"
        "5. 必要时建议用户咨询执业律师\n\n"
        "注意：你只能提供法律信息，不能替代执业律师的法律意见。"
    )
    model_type = ModelType.GPT_4O
