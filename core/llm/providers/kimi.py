"""
Kimi (Moonshot) Provider
预创建所有 Kimi 模型实例。
"""
from langchain_openai import ChatOpenAI

from core.llm.config import config, ProviderConfig

# 读取配置
_cfg = config.providers.get("kimi") or ProviderConfig()
_api_key = _cfg.api_key
_base_url = _cfg.base_url or "https://api.moonshot.cn/v1"


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
