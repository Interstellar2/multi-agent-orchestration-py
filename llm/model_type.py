"""
LLM 模型类型枚举
参考 evaluate_platform 的 LLMModelType，用 Enum 统一管理所有可用模型。
用户通过 ModelType 引用模型，而不是直接传字符串或 LLM 实例。
"""
from enum import Enum


class ModelType(str, Enum):
    """
    所有预置 LLM 模型。
    命名规范: {provider}-{model_name}
    """

    # OpenAI
    GPT_4O = "openai-gpt-4o"
    GPT_4O_MINI = "openai-gpt-4o-mini"
    GPT_4_TURBO = "openai-gpt-4-turbo"
    GPT_3_5_TURBO = "openai-gpt-3.5-turbo"

    # 阿里云百炼 (Bailian)
    QWEN_MAX = "bailian-qwen-max"
    QWEN_PLUS = "bailian-qwen-plus"
    QWEN_TURBO = "bailian-qwen-turbo"
    QWEN_3_5_PLUS = "bailian-qwen3.5-plus"
    QWEN_3_5_FLASH = "bailian-qwen3.5-flash"

    # DeepSeek
    DEEPSEEK_CHAT = "deepseek-chat"
    DEEPSEEK_REASONER = "deepseek-reasoner"

    # Kimi (Moonshot)
    KIMI_K2_5 = "kimi-kimi-k2.5"
    KIMI_K2 = "kimi-kimi-k2"
    KIMI_K1_5 = "kimi-kimi-k1.5"

    # OpenRouter
    OR_CLAUDE_3_5_SONNET = "openrouter-claude-3.5-sonnet"
    OR_GPT_4O = "openrouter-gpt-4o"
    OR_GPT_4O_MINI = "openrouter-gpt-4o-mini"
    OR_DEEPSEEK_CHAT = "openrouter-deepseek-chat"
