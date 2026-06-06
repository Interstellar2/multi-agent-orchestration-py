"""API 请求/响应模型"""
from typing import Literal, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., description="用户问题")
    mode: Literal["intent", "supervisor", "semantic"] = Field(default="semantic", description="路由模式")
    session_id: Optional[str] = Field(default=None, description="会话 ID（不传则新建单轮 Stateless）")
    stream: bool = Field(default=True, description="是否启用流式输出（暂未严格区分，默认 SSE 即流式）")


class ChatEvent(BaseModel):
    type: str = Field(..., description="事件类型: start/intent/semantic/retrieval/tool_call/tool_result/chunk/done/error")
    data: dict = Field(default_factory=dict, description="事件载荷")


class SessionCreateRequest(BaseModel):
    pass


class SessionResponse(BaseModel):
    session_id: str
    turn_count: int
    summary: Optional[str] = None
    turns: Optional[list] = None


class SessionListResponse(BaseModel):
    sessions: list[SessionResponse]
