"""
意图识别模块
用 LLM + 结构化输出识别用户意图。
"""
from typing import List, Optional

from langchain.schema import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from core.llm.factory import llm_factory
from core.llm.model_type import ModelType
from core.utils.logger import get_logger

logger = get_logger(__name__)


class IntentResult(BaseModel):
    intent: str = Field(description="识别的意图类别")
    confidence: float = Field(description="置信度", ge=0, le=1)
    reason: str = Field(description="分类理由")


class IntentClassifier:
    """
    意图识别器。
    支持预定义意图列表，也支持让模型自由推断。
    """

    def __init__(
        self,
        intents: List[str],
        intent_descriptions: Optional[dict] = None,
        model_type: ModelType = None,
        llm: Optional[BaseChatModel] = None,
    ):
        self.intents = intents
        self.intent_descriptions = intent_descriptions or {}
        if llm is not None:
            self._llm = llm
        else:
            self._llm = llm_factory.get_model(model_type or ModelType.GPT_4O_MINI)

    async def classify(self, query: str) -> IntentResult:
        logger.info(f"[意图识别] 开始 | query={query[:80]}")
        intent_list = "\n".join(
            f"- {i}: {self.intent_descriptions.get(i, 'No description')}"
            for i in self.intents
        )

        system_prompt = (
            "You are an intent recognition assistant. "
            "Analyze the user's input and classify it into one of the available intents.\n\n"
            f"Available intents:\n{intent_list}"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]

        structured_llm = self._llm.with_structured_output(IntentResult)
        result = await structured_llm.ainvoke(messages)
        logger.info(
            "[意图识别] 完成 | intent=%s confidence=%.2f reason=%s",
            result.intent,
            result.confidence,
            result.reason[:60],
        )
        return result
