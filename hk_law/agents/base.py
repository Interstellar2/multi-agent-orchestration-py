"""
香港法律 Agent 基类
继承自 agents.base.Agent，增加 RAG 检索能力。

每个子类对应一个法域，自带：
  - 法域名称 (domain)
  - 系统提示词 (system_prompt)
  - 向量检索器 (DomainRetriever)
"""
from typing import Any, Dict, Optional

from langchain.schema import Document, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from core.agents.base import Agent as BaseAgent
from hk_law.rag.engine import DomainRetriever
from core.llm.model_type import ModelType
from core.utils.logger import get_logger

logger = get_logger(__name__)


class HKLawAgent(BaseAgent):
    """
    香港法律 Agent 基类。

    子类需要设置：
      - name: Agent 标识
      - domain: 法域名称（对应 documents/ 和 vector_stores/ 目录）
      - system_prompt: 系统提示词
      - model_type: 默认模型
    """

    domain: str = ""
    _retriever: Optional[DomainRetriever] = None

    def __init__(
        self,
        model_type: ModelType = None,
        llm: Optional[BaseChatModel] = None,
        top_k: int = 5,
    ):
        super().__init__(model_type=model_type, llm=llm)
        if self.domain:
            self._retriever = DomainRetriever(domain=self.domain, top_k=top_k)

    async def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        执行流程：
        1. RAG 检索相关法律条文
        2. 将检索结果组装成上下文
        3. 调用 LLM 生成回答
        """
        logger.info(f"[{self.name}] 开始 | query={query[:80]}")

        # 1. 检索
        retrieved_docs: list[Document] = []
        if self._retriever:
            try:
                retrieved_docs = self._retriever.search(query)
                logger.info(f"[{self.name}] RAG 检索完成 | 召回 {len(retrieved_docs)} 条文档")
            except Exception as e:
                # ES 未启动、索引不存在、或网络问题
                logger.warning(f"[{self.name}] RAG 检索失败（将不使用检索上下文）: {e}")

        # 2. 组装上下文
        context_text = self._build_context(retrieved_docs)

        # 3. 调用 LLM
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(
                content=f"用户问题：{query}\n\n相关法律条文：\n{context_text}\n\n请根据上述法律条文回答用户问题。"
            ),
        ]
        try:
            response = await self._llm.ainvoke(messages)
            output = response.content
            logger.info(f"[{self.name}] 运行完成 | output_len={len(output)}")
            return output
        except Exception as e:
            logger.error(f"[{self.name}] LLM 调用失败: {e}")
            raise

    def _build_context(self, docs: list[Document]) -> str:
        """将检索到的文档组装成上下文字符串"""
        if not docs:
            return "（未能检索到相关法律条文）"

        parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "未知来源")
            parts.append(f"[{i}] 来源：{source}\n{doc.page_content.strip()}")

        return "\n\n".join(parts)

    def rebuild_index(self):
        """重新构建向量索引（文档更新后调用）"""
        if self._retriever:
            self._retriever.rebuild()
