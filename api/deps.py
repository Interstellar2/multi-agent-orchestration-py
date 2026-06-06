"""FastAPI 依赖注入"""
import os
from functools import lru_cache
from typing import Optional

from core.session import InMemorySessionStore, PostgresSessionStore, SessionStore


@lru_cache
def get_session_store() -> SessionStore:
    """
    获取 Session 存储实例。
    优先使用 PostgreSQL（如果 DATABASE_URL 环境变量存在），否则回退到内存存储。
    """
    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return PostgresSessionStore(dsn=dsn)
    return InMemorySessionStore()
