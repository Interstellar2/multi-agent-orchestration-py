"""
Session 存储层

提供多种持久化实现：内存、文件、PostgreSQL(SQLAlchemy ORM)。
所有实现遵循 SessionStore 抽象接口，上层无感知切换。
"""
import json
import os
from abc import ABC, abstractmethod
from typing import Optional

from core.session.models import Session, Turn
from core.utils.logger import get_logger

logger = get_logger(__name__)


class SessionStore(ABC):
    """Session 存储抽象接口。"""

    @abstractmethod
    async def get(self, session_id: str) -> Optional[Session]: ...

    @abstractmethod
    async def save(self, session: Session) -> None: ...

    @abstractmethod
    async def delete(self, session_id: str) -> None: ...


class InMemorySessionStore(SessionStore):
    """内存存储，进程退出丢失，适合测试和演示。"""

    def __init__(self):
        self._data: dict[str, Session] = {}

    async def get(self, session_id: str) -> Optional[Session]:
        return self._data.get(session_id)

    async def save(self, session: Session) -> None:
        self._data[session.session_id] = session

    async def delete(self, session_id: str) -> None:
        self._data.pop(session_id, None)


class FileSessionStore(SessionStore):
    """
    文件存储，每个 session 一个 JSON 文件。

    路径规则: {base_dir}/{session_id}.json
    """

    def __init__(self, base_dir: str = "./sessions"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def _path(self, session_id: str) -> str:
        return os.path.join(self.base_dir, f"{session_id}.json")

    async def get(self, session_id: str) -> Optional[Session]:
        path = self._path(session_id)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Session.model_validate(data)
        except Exception as e:
            logger.error(f"[FileSessionStore] 读取失败 | {session_id} | {e}")
            return None

    async def save(self, session: Session) -> None:
        path = self._path(session.session_id)
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(session.model_dump(mode="json"), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"[FileSessionStore] 写入失败 | {session.session_id} | {e}")

    async def delete(self, session_id: str) -> None:
        path = self._path(session_id)
        if os.path.exists(path):
            os.remove(path)


class PostgresSessionStore(SessionStore):
    """
    PostgreSQL 异步存储，基于 SQLAlchemy 2.0 Async ORM + asyncpg。

    用法:
        store = PostgresSessionStore()
        # 首次使用需运行 Alembic 迁移:
        #   alembic upgrade head
    """

    def __init__(self, dsn: Optional[str] = None):
        self.dsn = dsn

    def _session(self):
        """获取新的 AsyncSession。"""
        from core.session.db import get_session_maker

        return get_session_maker(self.dsn)()

    async def get(self, session_id: str) -> Optional[Session]:
        from core.session.db import SessionORM, TurnORM
        from sqlalchemy import select

        async with self._session() as db:
            db_session = await db.scalar(
                select(SessionORM).where(SessionORM.session_id == session_id)
            )
            if not db_session:
                return None

            turn_rows = await db.scalars(
                select(TurnORM)
                .where(TurnORM.session_id == session_id)
                .order_by(TurnORM.created_at)
            )

            turns = [
                Turn(
                    turn_id=r.turn_id,
                    role=r.role,
                    content=r.content,
                    metadata=r.metadata_ or {},
                    created_at=r.created_at,
                )
                for r in turn_rows
            ]

            return Session(
                session_id=session_id,
                created_at=db_session.created_at,
                updated_at=db_session.updated_at,
                turns=turns,
                summary=db_session.summary or "",
            )

    async def save(self, session: Session) -> None:
        from core.session.db import SessionORM, TurnORM
        from sqlalchemy import select

        async with self._session() as db:
            # upsert session
            existing = await db.scalar(
                select(SessionORM).where(SessionORM.session_id == session.session_id)
            )
            if existing:
                existing.updated_at = session.updated_at
                existing.summary = session.summary
            else:
                db.add(
                    SessionORM(
                        session_id=session.session_id,
                        created_at=session.created_at,
                        updated_at=session.updated_at,
                        summary=session.summary,
                    )
                )

            # 清旧 turns 再全量写入（简单策略）
            await db.execute(
                TurnORM.__table__.delete().where(TurnORM.session_id == session.session_id)
            )
            for t in session.turns:
                db.add(
                    TurnORM(
                        turn_id=t.turn_id,
                        session_id=session.session_id,
                        role=t.role,
                        content=t.content,
                        metadata_=t.metadata,
                        created_at=t.created_at,
                    )
                )

            await db.commit()

    async def delete(self, session_id: str) -> None:
        from core.session.db import SessionORM, TurnORM
        from sqlalchemy import delete

        async with self._session() as db:
            await db.execute(
                delete(TurnORM).where(TurnORM.session_id == session_id)
            )
            await db.execute(
                delete(SessionORM).where(SessionORM.session_id == session_id)
            )
            await db.commit()
