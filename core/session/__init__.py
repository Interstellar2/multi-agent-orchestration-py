"""
Session / Conversation 管理模块

提供多轮对话的持久化存储、历史窗口管理和法域锁定优化。

用法:
    from core.session import ConversationManager, PostgresSessionStore

    store = PostgresSessionStore(dsn="postgresql://...")
    manager = ConversationManager(store=store, max_turns=10)

    history = await manager.load_history(session_id)
    await manager.append_turn(session_id, "human", query)
    await manager.append_turn(session_id, "ai", output)
"""
from core.session.models import Session, Turn
from core.session.store import (
    SessionStore,
    InMemorySessionStore,
    FileSessionStore,
    PostgresSessionStore,
)
from core.session.manager import ConversationManager

__all__ = [
    "Session",
    "Turn",
    "SessionStore",
    "InMemorySessionStore",
    "FileSessionStore",
    "PostgresSessionStore",
    "ConversationManager",
]
