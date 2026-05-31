"""
阿里云百炼 (Bailian) Provider
预创建所有百炼模型实例。
"""
from langchain_openai import ChatOpenAI

from llm.config import config

# 读取配置
_cfg = config.providers.get("bailian", {})
_api_key = _cfg.api_key if _cfg else ""
_base_url = _cfg.base_url if _cfg else "https://dashscope.aliyuncs.com/compatible-mode/v1"


qwen_max = ChatOpenAI(
    model="qwen-max",
    api_key=_api_key,
    base_url=_base_url,
    temperature=0.7,
)

qwen_plus = ChatOpenAI(
    model="qwen-plus",
    api_key=_api_key,
    base_url=_base_url,
    temperature=0.7,
)

qwen_turbo = ChatOpenAI(
    model="qwen-turbo",
    api_key=_api_key,
    base_url=_base_url,
    temperature=0.7,
)

qwen_3_5_plus = ChatOpenAI(
    model="qwen3.5-plus",
    api_key=_api_key,
    base_url=_base_url,
    temperature=0.7,
    model_kwargs={"extra_body": {"enable_thinking": False}},
)

qwen_3_5_flash = ChatOpenAI(
    model="qwen3.5-flash",
    api_key=_api_key,
    base_url=_base_url,
    temperature=0.7,
    model_kwargs={"extra_body": {"enable_thinking": False}},
)
