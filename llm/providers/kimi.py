"""
Kimi (Moonshot) Provider
预创建所有 Kimi 模型实例。
"""
from langchain_openai import ChatOpenAI

from llm.config import config

# 读取配置
_cfg = config.providers.get("kimi", {})
_api_key = _cfg.api_key if _cfg else ""
_base_url = _cfg.base_url if _cfg else "https://api.moonshot.cn/v1"


kimi_k2_5 = ChatOpenAI(
    model="kimi-k2.5",
    api_key=_api_key,
    base_url=_base_url,
    temperature=0.7,
)

kimi_k2 = ChatOpenAI(
    model="kimi-k2",
    api_key=_api_key,
    base_url=_base_url,
    temperature=0.7,
)

kimi_k1_5 = ChatOpenAI(
    model="kimi-k1.5",
    api_key=_api_key,
    base_url=_base_url,
    temperature=0.7,
)
