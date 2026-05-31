"""
公司法域 Agent
处理公司设立、公司治理、股东权益、清盘等法律问题。
"""
from core.llm.model_type import ModelType
from hk_law.agents.base import HKLawAgent


class CompanyLawAgent(HKLawAgent):
    """公司法 Agent"""
    name = "company"
    domain = "company"
    system_prompt = (
        "你是一名香港公司法律专家，精通《公司条例》(第622章)及相关法规。你的职责是：\n"
        "1. 解答公司设立、注册及合规问题\n"
        "2. 分析董事责任及股东权利\n"
        "3. 解释公司清盘及重组程序\n"
        "4. 就上市公司的披露义务提供指引\n"
        "5. 必要时建议用户咨询执业律师或公司秘书\n\n"
        "注意：你只能提供法律信息，不能替代执业律师的法律意见。"
    )
    model_type = ModelType.GPT_4O
