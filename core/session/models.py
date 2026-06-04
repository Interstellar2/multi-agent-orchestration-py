"""
Session / Turn 数据模型

定义多轮对话的核心数据结构，兼容 Pydantic v2 验证。
"""
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Turn(BaseModel):
    """
    单条对话记录（一轮交互中的一个片段）。

    一个用户提问 -> Agent 回答 构成一轮，但 ReAct 内部可能有多个 tool-call
    往返，每个往返都作为一个 Turn 记录。
    """

    turn_id: str = Field(default_factory=lambda: str(uuid4()))
    role: Literal["human", "ai", "tool", "system"]
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Session(BaseModel):
    """
    一个完整的多轮对话会话。

    - turns: 按时间顺序排列的完整原始记录
    - summary: 早期轮次的压缩摘要（由 ConversationManager 自动生成）
    """

    session_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    turns: List[Turn] = Field(default_factory=list)
    summary: str = ""

    def model_post_init(self, __context: Any) -> None:
        """确保 updated_at 初始化与 created_at 一致。"""
        if self.updated_at is None:
            self.updated_at = self.created_at
