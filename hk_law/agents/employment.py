"""
雇佣法域 Agent
处理劳动合同、解雇、遣散费、歧视等雇佣法律问题。
"""
from core.llm.model_type import ModelType
from hk_law.agents.base import HKLawAgent


class EmploymentLawAgent(HKLawAgent):
    """雇佣法 Agent"""
    name = "employment"
    domain = "employment"
    system_prompt = (
        "你是一名香港雇佣法律专家，精通《雇佣条例》(第57章)、《最低工资条例》(第608章) "
        "及《性别歧视条例》(第480章)等相关法规。你的职责是：\n"
        "1. 解答劳动合同、试用期及通知期的法律问题\n"
        "2. 计算法定权益（如年假、遣散费、长期服务金）\n"
        "3. 分析不当解雇及歧视申索的程序\n"
        "4. 解释劳工处的调解及劳资审裁处的程序\n"
        "5. 必要时建议用户咨询执业律师或劳工处\n\n"
        "注意：你只能提供法律信息，不能替代执业律师的法律意见。"
    )
    model_type = ModelType.GPT_4O
