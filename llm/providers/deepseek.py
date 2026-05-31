"""
DeepSeek Provider
预创建所有 DeepSeek 模型实例。
"""
from langchain_openai import ChatOpenAI

from llm.config import config

# 读取配置
_cfg = config.providers.get("deepseek", {})
_api_key = _cfg.api_key if _cfg else ""
_base_url = _cfg.base_url if _cfg else "https://api.deepseek.com/v1"


deepseek_chat = ChatOpenAI(
    model="deepseek-chat",
    api_key=_api_key,
    base_url=_base_url,
    temperature=0.7,
)

deepseek_reasoner = ChatOpenAI(
    model="deepseek-reasoner",
    api_key=_api_key,
    base_url=_base_url,
    temperature=0.7,
)
