"""
预置的子 Agent 示例。
添加新 Agent 只需：
1. 继承 Agent 基类
2. 设置 name 和 system_prompt
3. （可选）设置默认 model_type
4. （可选）重写 run 方法实现自定义逻辑
"""
from typing import Any, Dict, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from core.agents.base import Agent
from core.llm.model_type import ModelType


class SearchAgent(Agent):
    """搜索/信息查询 Agent"""
    name = "search"
    system_prompt = (
        "You are a search assistant. Help the user find information, "
        "summarize key points clearly, and cite sources when possible."
    )
    model_type = ModelType.GPT_4O_MINI


class CodeAgent(Agent):
    """编程/代码 Agent"""
    name = "code"
    system_prompt = (
        "You are a coding assistant. Write clean, well-commented code, "
        "explain your approach, and handle debugging requests."
    )
    model_type = ModelType.GPT_4O


class ChatAgent(Agent):
    """闲聊/通用问答 Agent"""
    name = "chat"
    system_prompt = (
        "You are a friendly conversational assistant. Answer in a natural, "
        "helpful tone. If the user asks something beyond general chat, "
        "suggest they use a specialized agent."
    )
    model_type = ModelType.GPT_4O_MINI


class AnalysisAgent(Agent):
    """数据分析 Agent（演示扩展性）"""
    name = "analysis"
    system_prompt = (
        "You are a data analysis assistant. Help interpret data, "
        "suggest visualizations, and explain statistical concepts."
    )
    model_type = ModelType.GPT_4O

    async def run(self, query: str, context: Dict[str, Any] = None) -> str:
        # 自定义逻辑示例：假装调用了一个数据分析工具
        base = await super().run(query, context)
        return f"[Analysis Result]\n{base}"


# 注册表，方便通过名称获取 Agent
_AGENT_REGISTRY = {
    agent.name: agent
    for agent in [SearchAgent, CodeAgent, ChatAgent, AnalysisAgent]
}


def get_agent(
    name: str,
    model_type: ModelType = None,
    llm: Optional[BaseChatModel] = None,
) -> Agent:
    """通过名称实例化 Agent，自动传递 LLM 配置"""
    agent_cls = _AGENT_REGISTRY.get(name)
    if not agent_cls:
        raise ValueError(
            f"Unknown agent: {name}. Available: {list(_AGENT_REGISTRY.keys())}"
        )
    return agent_cls(model_type=model_type, llm=llm)
