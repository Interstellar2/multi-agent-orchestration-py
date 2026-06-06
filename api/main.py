"""FastAPI 应用入口

启动方式:
    uvicorn api.main:app --reload --port 8000

或:
    python -m uvicorn api.main:app --reload --port 8000
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.routers import chat, session


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期事件"""
    # 启动时：可以预热连接池、检查 ES 等
    yield
    # 关闭时：清理资源


app = FastAPI(
    title="Multi-Agent API",
    description="香港法律多 Agent 系统 Web API（SSE 流式输出）",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS：允许前端独立开发时跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(session.router, prefix="/api", tags=["session"])


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/")
async def serve_index():
    return FileResponse("web/index.html")
