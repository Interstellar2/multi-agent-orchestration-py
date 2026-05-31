"""
LLM 工厂
参考 evaluate_platform 的 LLMFactory，统一管理所有 LLM 模型实例。

用法:
    from llm.factory import llm_factory
    from llm.model_type import ModelType

    model = llm_factory.get_model(ModelType.GPT_4O_MINI)
    # 或
    model = llm_factory.get_model("openai-gpt-4o-mini")

    # 动态注册新模型
    llm_factory.register_model(ModelType.GPT_4O, my_custom_llm)
"""
from typing import Dict, List, Union

from langchain_core.language_models.chat_models import BaseChatModel

from core.llm.model_type import ModelType
from core.llm.providers import openai, bailian, deepseek, kimi, openrouter
from core.utils.logger import get_logger

logger = get_logger(__name__)


class LLMFactory:
    """LLM 工厂，用于管理和切换不同的 LLM 模型"""

    def __init__(self):
        self._models: Dict[ModelType, BaseChatModel] = {
            # OpenAI
            ModelType.GPT_4O: openai.gpt_4o,
            ModelType.GPT_4O_MINI: openai.gpt_4o_mini,
            ModelType.GPT_4_TURBO: openai.gpt_4_turbo,
            ModelType.GPT_3_5_TURBO: openai.gpt_3_5_turbo,
            # 百炼
            ModelType.QWEN_MAX: bailian.qwen_max,
            ModelType.QWEN_PLUS: bailian.qwen_plus,
            ModelType.QWEN_TURBO: bailian.qwen_turbo,
            ModelType.QWEN_3_5_PLUS: bailian.qwen_3_5_plus,
            ModelType.QWEN_3_5_FLASH: bailian.qwen_3_5_flash,
            # DeepSeek
            ModelType.DEEPSEEK_CHAT: deepseek.deepseek_chat,
            ModelType.DEEPSEEK_REASONER: deepseek.deepseek_reasoner,
            # Kimi
            ModelType.KIMI_K2_5: kimi.kimi_k2_5,
            ModelType.KIMI_K2: kimi.kimi_k2,
            ModelType.KIMI_K1_5: kimi.kimi_k1_5,
            # OpenRouter
            ModelType.OR_CLAUDE_3_5_SONNET: openrouter.openrouter_claude_3_5_sonnet,
            ModelType.OR_GPT_4O: openrouter.openrouter_gpt_4o,
            ModelType.OR_GPT_4O_MINI: openrouter.openrouter_gpt_4o_mini,
            ModelType.OR_DEEPSEEK_CHAT: openrouter.openrouter_deepseek_chat,
        }

    def get_model(self, model_type: Union[str, ModelType]) -> BaseChatModel:
        """通过 ModelType 或字符串获取模型实例"""
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type)
            except ValueError as e:
                logger.error(f"不支持的模型类型: {model_type}")
                raise ValueError(
                    f"Unsupported model type: {model_type}. "
                    f"Available: {[m.value for m in ModelType]}"
                ) from e

        if model_type not in self._models:
            logger.error(f"模型未配置: {model_type.value}")
            raise ValueError(
                f"Model {model_type.value} is not configured. "
                f"Available types: {[m.value for m in self._models.keys()]}"
            )

        logger.debug(f"[LLMFactory] 获取模型: {model_type.value}")
        return self._models[model_type]

    def register_model(self, model_type: Union[str, ModelType], model: BaseChatModel) -> None:
        """动态注册新模型"""
        if isinstance(model_type, str):
            try:
                model_type = ModelType(model_type)
            except ValueError as e:
                raise ValueError(f"Unsupported model type: {model_type}") from e

        self._models[model_type] = model

    def list_models(self) -> List[str]:
        """列出所有已注册模型"""
        return [m.value for m in self._models.keys()]


# 全局单例
llm_factory = LLMFactory()
