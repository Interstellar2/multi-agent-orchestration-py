"""
OpenAI Provider
预创建所有 OpenAI 兼容模型实例。
"""
from langchain_openai import ChatOpenAI

from core.llm.config import config, ProviderConfig

# 读取配置
_openai_cfg = config.providers.get("openai") or ProviderConfig()
_openai_api_key = _openai_cfg.api_key
_openai_base_url = _openai_cfg.base_url or "https://api.openai.com/v1"


# 预创建模型实例
gpt_4o = ChatOpenAI(
    model="gpt-4o",
    api_key=_openai_api_key,
    base_url=_openai_base_url,
    temperature=0.7,
)

gpt_4o_mini = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=_openai_api_key,
    base_url=_openai_base_url,
    temperature=0.7,
)

gpt_4_turbo = ChatOpenAI(
    model="gpt-4-turbo-preview",
    api_key=_openai_api_key,
    base_url=_openai_base_url,
    temperature=0.7,
)

gpt_3_5_turbo = ChatOpenAI(
    model="gpt-3.5-turbo",
    api_key=_openai_api_key,
    base_url=_openai_base_url,
    temperature=0.7,
)
