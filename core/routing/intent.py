"""
意图识别模块
用 LLM + 结构化输出识别用户意图。
"""
from typing import List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
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


class LegalQueryAnalysis(BaseModel):
    """
    法律查询结构化分析结果。
    将用户口语化的问题解析为法域、法条、改写查询等多维度信息，
    用于精准路由和 RAG 检索优化。
    """

    intent: str = Field(
        description="用户核心意图类别。可选：法律咨询、程序指引、权益计算、比较分析、风险评估、其他"
    )
    jurisdictions: List[str] = Field(
        description="涉及的法域列表（如 ['criminal', 'employment']）。问题可能涉及多个法域。"
    )
    statutes: List[str] = Field(
        description="涉及的具体香港法例名称，如 ['雇佣条例 第57章', '性别歧视条例 第480章']。如无法确定可留空。"
    )
    rewritten_query: str = Field(
        description="改写后的查询：去除口语化表达，补充法律术语和同义词，使其更适合向量检索。保留原始关键事实（时间、金额、当事人关系等）。"
    )
    is_cross_domain: bool = Field(
        description="是否涉及跨法域问题（如同时涉及雇佣和民事）"
    )
    confidence: float = Field(description="整体置信度", ge=0, le=1)
    reason: str = Field(description="分析理由，简要说明为何归入这些法域和法条")


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


class LegalQueryAnalyzer:
    """
    法律查询语义分析器。

    将用户的口语化法律问题解析为多维度结构化信息：
      - 法域归属（支持多标签）
      - 涉及法条
      - 语义改写（优化向量检索）
      - 是否跨域
      - 核心意图

    用法:
        analyzer = LegalQueryAnalyzer(domains=["criminal", "civil", ...])
        analysis = await analyzer.analyze("我被公司无故解雇怎么办？")
        # analysis.jurisdictions -> ["employment"]
        # analysis.rewritten_query -> "不当解雇 雇佣条例 第57章 遣散费 赔偿..."
    """

    def __init__(
        self,
        domains: List[str],
        domain_descriptions: Optional[dict] = None,
        model_type: ModelType = None,
        llm: Optional[BaseChatModel] = None,
    ):
        self.domains = domains
        self.domain_descriptions = domain_descriptions or {}
        if llm is not None:
            self._llm = llm
        else:
            self._llm = llm_factory.get_model(model_type or ModelType.GPT_4O_MINI)

    async def analyze(self, query: str) -> LegalQueryAnalysis:
        logger.info(f"[语义分析] 开始 | query={query[:80]}")

        domain_list = "\n".join(
            f"- {d}: {self.domain_descriptions.get(d, 'No description')}"
            for d in self.domains
        )

        system_prompt = (
            "你是一名香港法律查询语义分析专家。你的任务是将用户的口语化法律问题解析为结构化信息，"
            "用于精准路由到对应法域 Agent 并优化向量检索。\n\n"
            "分析要求：\n"
            "1. jurisdictions: 从下方法域列表中选出所有相关法域（多标签）。"
            "如果问题同时涉及多个法域，全部列出。\n"
            "2. statutes: 尽可能列出具体涉及的香港法例名称及章号，如不确定可留空列表。\n"
            "3. rewritten_query: 将用户问题改写为更适合向量检索的正式表述。"
            "保留所有关键事实（时间、金额、当事人关系），替换口语化为法律术语，"
            "补充相关同义词。例如 '被老板炒了' -> '不当解雇、无理解雇、终止雇佣关系'。\n"
            "4. is_cross_domain: 当涉及多个法域时为 true。\n"
            "5. intent: 判断用户核心诉求（法律咨询 / 程序指引 / 权益计算 / 比较分析 / 风险评估 / 其他）。\n\n"
            f"可用法域：\n{domain_list}\n\n"
            "重要：必须以 LegalQueryAnalysis 的 JSON Schema 返回结果。"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=query),
        ]

        structured_llm = self._llm.with_structured_output(LegalQueryAnalysis)
        result = await structured_llm.ainvoke(messages)

        # 校验：确保返回的 jurisdictions 都在可用列表内
        valid_jurisdictions = [d for d in result.jurisdictions if d in self.domains]
        if not valid_jurisdictions and self.domains:
            valid_jurisdictions = [self.domains[0]]
            result.is_cross_domain = False

        result.jurisdictions = valid_jurisdictions

        logger.info(
            "[语义分析] 完成 | intent=%s jurisdictions=%s cross_domain=%s confidence=%.2f",
            result.intent,
            result.jurisdictions,
            result.is_cross_domain,
            result.confidence,
        )
        return result
