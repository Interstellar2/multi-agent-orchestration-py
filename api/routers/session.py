"""Session 管理端点"""
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException

from api.deps import get_session_store
from api.schemas import SessionListResponse, SessionResponse
from core.session import SessionStore

router = APIRouter()


@router.get("/sessions", response_model=SessionListResponse)
async def list_sessions(store: SessionStore = Depends(get_session_store)):
    """列出所有活跃会话（仅支持内存/PostgreSQL 存储）。"""
    sessions = await store.list_all()
    items: List[SessionResponse] = []
    for s in sessions:
        items.append(
            SessionResponse(
                session_id=s.session_id,
                turn_count=len(s.turns),
                summary=s.summary,
            )
        )
    return SessionListResponse(sessions=items)


@router.get("/sessions/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str, store: SessionStore = Depends(get_session_store)):
    """获取单个会话详情（含完整对话记录）。"""
    session = await store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return SessionResponse(
        session_id=session.session_id,
        turn_count=len(session.turns),
        summary=session.summary,
        turns=[t.model_dump(mode="json") for t in session.turns],
    )


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, store: SessionStore = Depends(get_session_store)):
    """删除指定会话。"""
    await store.delete(session_id)
    return {"ok": True, "session_id": session_id}
