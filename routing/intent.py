"""
意图识别模块
用 LLM + 结构化输出识别用户意图。
"""
from typing import List, Optional

from langchain.schema import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field

from llm.factory import llm_factory
from llm.model_type import ModelType


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
        return await structured_llm.ainvoke(messages)
