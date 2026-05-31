"""
OpenRouter Provider
预创建常用 OpenRouter 模型实例。
OpenRouter 是统一 API 网关，支持调用多家模型。

注意：OpenRouter 建议传入 default_headers 以标识应用。
"""
from langchain_openai import ChatOpenAI

from llm.config import config

# 读取配置
_cfg = config.providers.get("openrouter", {})
_api_key = _cfg.api_key if _cfg else ""
_base_url = _cfg.base_url if _cfg else "https://openrouter.ai/api/v1"

# OpenRouter 推荐传入应用标识头（可选）
_default_headers = {}
if _cfg and getattr(_cfg, "app_name", None):
    _default_headers["X-Title"] = _cfg.app_name
    _default_headers["HTTP-Referer"] = getattr(_cfg, "app_url", "")


openrouter_claude_3_5_sonnet = ChatOpenAI(
    model="anthropic/claude-3.5-sonnet",
    api_key=_api_key,
    base_url=_base_url,
    temperature=0.7,
    default_headers=_default_headers,
)

openrouter_gpt_4o = ChatOpenAI(
    model="openai/gpt-4o",
    api_key=_api_key,
    base_url=_base_url,
    temperature=0.7,
    default_headers=_default_headers,
)

openrouter_gpt_4o_mini = ChatOpenAI(
    model="openai/gpt-4o-mini",
    api_key=_api_key,
    base_url=_base_url,
    temperature=0.7,
    default_headers=_default_headers,
)

openrouter_deepseek_chat = ChatOpenAI(
    model="deepseek/deepseek-chat",
    api_key=_api_key,
    base_url=_base_url,
    temperature=0.7,
    default_headers=_default_headers,
)
