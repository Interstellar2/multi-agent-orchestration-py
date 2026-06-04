"""
ConversationManager

多轮对话管理器：
  - 历史窗口管理（最近 N 轮完整保留，更早的摘要）
  - Fast-Path 法域锁定（连续同法域追问优化）
  - Turn 持久化读写

用法:
    manager = ConversationManager(store=PostgresSessionStore(), max_turns=10)
    history = await manager.load_history(session_id)
    await manager.append_turn(session_id, "human", query, metadata={"mode": "intent"})
    await manager.append_turn(session_id, "ai", output, metadata={"domain": "employment"})

    # Fast-Path 检查
    fast_domain = manager.check_fast_path(session_id, current_query)
"""
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from core.session.models import Session, Turn
from core.session.store import SessionStore
from core.utils.logger import get_logger

logger = get_logger(__name__)


class ConversationManager:
    """
    多轮对话管理器。

    Args:
        store: Session 持久化存储
        max_turns: 完整保留的最近轮次数（每轮可能含多个 Turn）
        max_tokens: 触发摘要的 token 阈值（预留，暂未严格计算）
    """

    def __init__(
        self,
        store: SessionStore,
        max_turns: int = 10,
        max_tokens: int = 8000,
    ):
        self.store = store
        self.max_turns = max_turns
        self.max_tokens = max_tokens

    # ------------------------------------------------------------------
    # 历史加载
    # ------------------------------------------------------------------

    async def load_history(self, session_id: str) -> List[BaseMessage]:
        """
        加载适合传给 LLM 的 BaseMessage 列表。

        策略：
          1. 总 Turn 数 <= max_turns：全部保留原始 messages
          2. 总 Turn 数 > max_turns：老轮次压缩为 summary，新轮次保留完整
        """
        session = await self.store.get(session_id)
        if not session:
            return []

        turns = session.turns
        if len(turns) <= self.max_turns:
            return self._turns_to_messages(turns)

        # 需要摘要：老轮次压缩，新轮次保留
        old_turns = turns[: -self.max_turns]
        recent_turns = turns[-self.max_turns :]

        if not session.summary:
            session.summary = await self._summarize(old_turns)
            session.updated_at = datetime.utcnow()
            await self.store.save(session)
            logger.info(
                f"[ConversationManager] 生成历史摘要 | session={session_id} "
                f"old_turns={len(old_turns)} recent={len(recent_turns)}"
            )

        summary_msg = SystemMessage(
            content=f"以下是对话历史摘要（较早的轮次已压缩）：\n{session.summary}"
        )
        return [summary_msg] + self._turns_to_messages(recent_turns)

    @staticmethod
    def _turns_to_messages(turns: List[Turn]) -> List[BaseMessage]:
        """将 Turn 列表转换为 LangChain BaseMessage 列表。"""
        messages: List[BaseMessage] = []
        for t in turns:
            if t.role == "human":
                messages.append(HumanMessage(content=t.content))
            elif t.role == "ai":
                messages.append(AIMessage(content=t.content))
            elif t.role == "tool":
                # tool 结果用 SystemMessage 包装，与 ReAct 内部一致
                messages.append(SystemMessage(content=t.content))
            elif t.role == "system":
                messages.append(SystemMessage(content=t.content))
        return messages

    # ------------------------------------------------------------------
    # 摘要生成
    # ------------------------------------------------------------------

    async def _summarize(self, turns: List[Turn]) -> str:
        """将早期 Turn 列表压缩为摘要文本。"""
        if not turns:
            return ""

        # 简单规则摘要：提取每轮 human/ai 的核心内容
        # 生产环境可替换为 LLM 调用生成更自然的摘要
        lines = []
        current_query = ""
        for t in turns:
            if t.role == "human":
                current_query = t.content[:200]
            elif t.role == "ai" and current_query:
                meta = t.metadata
                domain = meta.get("domain", "unknown")
                lines.append(f"- 用户询问（{domain}）: {current_query}")
                answer_preview = t.content[:200].replace("\n", " ")
                lines.append(f"  回答摘要: {answer_preview}...")
                current_query = ""
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Turn 写入
    # ------------------------------------------------------------------

    async def append_turn(
        self,
        session_id: str,
        role: Literal["human", "ai", "tool", "system"],
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """向指定 session 追加一条 Turn，自动创建不存在的 session。"""
        session = await self.store.get(session_id)
        if session is None:
            session = Session(session_id=session_id)
            logger.info(f"[ConversationManager] 创建新 session | {session_id}")

        session.turns.append(
            Turn(role=role, content=content, metadata=metadata or {})
        )
        session.updated_at = datetime.utcnow()
        await self.store.save(session)

    # ------------------------------------------------------------------
    # Fast-Path 法域锁定
    # ------------------------------------------------------------------

    def check_fast_path(
        self,
        session_id: str,
        query: str,
        min_confidence: float = 0.85,
        max_query_len: int = 20,
    ) -> Optional[str]:
        """
        Fast-Path 法域锁定。

        当检测到连续追问时，直接复用上轮法域，跳过 LLM 意图识别。

        判定条件（全部满足）：
          1. 最近 2 轮 user->assistant 的法域相同
          2. 上轮置信度 >= min_confidence
          3. 当前 query 长度 <= max_query_len 或 以追问词开头
          4. 无显式话题切换词

        Args:
            session_id: 当前会话 ID
            query: 用户当前输入
            min_confidence: 置信度阈值
            max_query_len: 短句长度阈值

        Returns:
            锁定的法域名称，或 None（不走 Fast-Path）
        """
        import asyncio

        # 需要用 await 获取 session，但本方法设计为同步（检查逻辑很轻）
        # 调用方应在外部 await store.get 后传入 session，或自行处理
        raise RuntimeError(
            "check_fast_path 已改为异步版本 check_fast_path_async，请使用新方法"
        )

    async def check_fast_path_async(
        self,
        session_id: str,
        query: str,
        min_confidence: float = 0.85,
        max_query_len: int = 20,
    ) -> Optional[str]:
        """
        Fast-Path 法域锁定（异步版本）。

        返回锁定的法域名称，或 None 表示不走 Fast-Path。
        """
        session = await self.store.get(session_id)
        if not session or len(session.turns) < 2:
            return None

        # 话题切换词黑名单
        switch_keywords = ("换个话题", "换一下", "另外", "再说说", "那刑法", "那民法", "那公司")
        if any(kw in query for kw in switch_keywords):
            return None

        # 追问词白名单（满足其一即可放宽长度限制）
        follow_up_prefixes = ("那", "还有", "能不能", "能不能", "详细", "具体", "为什么", "多少")
        is_short = len(query) <= max_query_len
        is_follow_up = any(query.startswith(p) for p in follow_up_prefixes)
        if not (is_short or is_follow_up):
            return None

        # 取最近 2 轮 AI Turn 的 metadata，看法域是否一致
        ai_turns = [t for t in reversed(session.turns) if t.role == "ai"]
        if len(ai_turns) < 2:
            return None

        domain_1 = ai_turns[0].metadata.get("domain")
        domain_2 = ai_turns[1].metadata.get("domain")
        conf_1 = ai_turns[0].metadata.get("confidence", 0.0)
        conf_2 = ai_turns[1].metadata.get("confidence", 0.0)

        if (
            domain_1
            and domain_1 == domain_2
            and conf_1 >= min_confidence
            and conf_2 >= min_confidence
        ):
            logger.info(
                f"[FastPath] 法域锁定 | session={session_id} domain={domain_1} "
                f"conf=({conf_1:.2f}, {conf_2:.2f}) query_len={len(query)}"
            )
            return domain_1

        return None

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    async def get_last_domain(self, session_id: str) -> Optional[str]:
        """获取最近一轮 AI 回答的法域（用于意图识别 fallback）。"""
        session = await self.store.get(session_id)
        if not session:
            return None
        for t in reversed(session.turns):
            if t.role == "ai":
                return t.metadata.get("domain")
        return None

    async def clear(self, session_id: str) -> None:
        """清空指定 session（/new 命令用）。"""
        await self.store.delete(session_id)
        logger.info(f"[ConversationManager] 清空 session | {session_id}")
