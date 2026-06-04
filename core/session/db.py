"""
SQLAlchemy ORM 模型 — Session / Turn 存储

使用 SQLAlchemy 2.0 Async API，驱动 asyncpg。
"""
from datetime import datetime, timezone
from typing import AsyncGenerator, Optional
from uuid import uuid4

from sqlalchemy import JSON, DateTime, String, Text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from core.utils.logger import get_logger

logger = get_logger(__name__)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class SessionORM(Base):
    """会话表"""

    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)
    summary: Mapped[str] = mapped_column(Text, default="")


class TurnORM(Base):
    """对话轮次表"""

    __tablename__ = "turns"

    turn_id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid4()))
    session_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_: Mapped[Optional[dict]] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


# ------------------------------------------------------------------
# Engine / Session Factory
# ------------------------------------------------------------------

_engine = None
_async_session_maker = None


def get_engine(dsn: Optional[str] = None):
    """获取或创建 async engine（单例）。"""
    global _engine
    if _engine is None:
        import os

        dsn = dsn or os.getenv(
            "DATABASE_URL",
            "postgresql+asyncpg://postgres:postgres@localhost:5432/multi_agent",
        )
        _engine = create_async_engine(dsn, echo=False, future=True)
        logger.info(f"[DB] Engine created | {dsn.split('@')[-1]}")
    return _engine


def get_session_maker(dsn: Optional[str] = None):
    """获取 async session maker（单例）。"""
    global _async_session_maker
    if _async_session_maker is None:
        _async_session_maker = async_sessionmaker(
            bind=get_engine(dsn),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _async_session_maker


async def get_session(dsn: Optional[str] = None) -> AsyncGenerator[AsyncSession, None]:
    """FastAPI 风格的依赖注入生成器。"""
    async with get_session_maker(dsn)() as session:
        yield session
