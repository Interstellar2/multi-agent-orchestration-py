"""聊天 SSE 端点"""
import asyncio
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends
from sse_starlette.sse import EventSourceResponse

from api.deps import get_session_store
from api.schemas import ChatRequest
from core.llm.model_type import ModelType
from core.session import SessionStore
from domains.hk_law.main import HKLawSystem

router = APIRouter()


async def _chat_event_generator(
    system: HKLawSystem,
    query: str,
    mode: str,
    session_id: str,
) -> AsyncGenerator[dict, None]:
    """
    将 DomainSystem.ask 的内部事件桥接到 SSE。
    使用 asyncio.Queue 解耦 producer（ask 回调）和 consumer（SSE 生成器）。
    """
    queue: asyncio.Queue[dict] = asyncio.Queue()
    task_exception: list = []

    async def on_event(evt: dict) -> None:
        await queue.put(evt)

    async def run_ask() -> None:
        try:
            await system.ask(
                query=query,
                mode=mode,
                session_id=session_id,
                event_callback=on_event,
            )
        except Exception as exc:
            task_exception.append(exc)
            await queue.put({"type": "error", "data": {"error": str(exc)}})
        finally:
            await queue.put(None)  # sentinel

    # 在后台运行 ask
    ask_task = asyncio.create_task(run_ask())

    try:
        while True:
            evt = await queue.get()
            if evt is None:
                break
            yield evt
            if evt.get("type") in ("done", "error"):
                break
    finally:
        ask_task.cancel()
        try:
            await ask_task
        except asyncio.CancelledError:
            pass


@router.post("/chat")
async def chat(
    req: ChatRequest,
    store: SessionStore = Depends(get_session_store),
):
    """
    流式聊天接口（SSE）。

    推送的事件类型:
      - start:   开始处理
      - intent:  意图识别结果（intent 模式）
      - semantic: 语义分析结果（semantic 模式）
      - fastpath: Fast-Path 法域锁定
      - retrieval: RAG 检索结果
      - tool_call: ReAct 调用工具
      - tool_result: 工具返回结果
      - chunk:   LLM 流式输出片段（后续可接入流式 LLM）
      - done:    完成（含最终 output 前 500 字摘要）
      - error:   错误
    """
    session_id = req.session_id or f"stateless-{uuid.uuid4().hex[:8]}"
    system = HKLawSystem(model_type=ModelType.GPT_4O_MINI, session_store=store)

    return EventSourceResponse(
        _chat_event_generator(
            system=system,
            query=req.query,
            mode=req.mode,
            session_id=session_id,
        ),
        media_type="text/event-stream",
    )
