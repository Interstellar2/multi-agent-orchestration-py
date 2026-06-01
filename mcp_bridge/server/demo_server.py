"""
MCP Demo Server
===============
一个用于演示 MCP 协议集成的轻量级 Server。

参考生产级 MCP Server（如 hk-finance-mcp）的架构模式：
- 基于 FastMCP 暴露结构化工具
- 工具输入/输出均经过 Pydantic 风格校验（由 MCP 协议层保证）
- 支持 stdio / sse 两种 transport，方便本地调试和远程接入

工具列表:
    - get_weather   : Mock 天气查询（无外部依赖，零敏感信息）
    - calculate     : 安全数学表达式计算（白名单 AST 解析，无 eval 风险）
    - search_docs   : 模拟文档检索（返回预设的演示片段）

用法:
    # stdio 模式（供本地 Agent 子进程调用）
    python -m mcp_bridge.server.demo_server --transport stdio

    # sse 模式（供远程 Agent HTTP 连接）
    python -m mcp_bridge.server.demo_server --transport sse --port 18080
"""
import argparse
import ast
import operator
from typing import Any, Optional

# 先加载项目统一日志配置，接管 mcp/fastmcp 库的日志输出
from core.utils.logger import get_logger

logger = get_logger("mcp_bridge.server.demo_server")

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("DemoMCPServer")

# ---------------------------------------------------------------------------
# 工具实现
# ---------------------------------------------------------------------------

@mcp.tool()
def get_weather(location: str) -> str:
    """
    查询指定城市的天气情况。

    Args:
        location: 城市名称，例如 "Beijing", "Hong Kong"。

    Returns:
        天气描述字符串。
    """
    logger.info(f"[get_weather] 调用 | location={location}")
    # 纯 Mock 数据，无外部 API 调用，零业务敏感信息
    mock_db = {
        "beijing": "北京今天晴朗，气温 26°C，空气质量优。",
        "hong kong": "香港今天多云，气温 29°C，湿度 78%。",
        "shanghai": "上海今天小雨，气温 22°C，建议带伞。",
        "shenzhen": "深圳今天晴间多云，气温 30°C。",
        "tokyo": "东京今天阴，气温 24°C，偶有阵雨。",
    }
    key = location.strip().lower()
    result = mock_db.get(key, f"{location} 今天天气不错，气温约 25°C。（提示：这是演示数据）")
    logger.info(f"[get_weather] 返回 | {result}")
    return result


# 安全计算白名单
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def _safe_eval(node: ast.AST) -> Any:
    """递归解析 AST，仅允许常量与基本四则运算。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"不支持的运算符: {op_type.__name__}")
        return _SAFE_OPS[op_type](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"不支持的运算符: {op_type.__name__}")
        return _SAFE_OPS[op_type](_safe_eval(node.operand))
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    raise ValueError(f"不支持的表达式节点: {type(node).__name__}")


@mcp.tool()
def calculate(expression: str) -> str:
    """
    安全计算数学表达式。

    支持的运算符: +, -, *, /, //, %, **, 括号。
    不支持函数调用、变量、属性访问等复杂语法。

    Args:
        expression: 数学表达式字符串，例如 "(3 + 5) * 2"。

    Returns:
        计算结果字符串，或错误提示。
    """
    logger.info(f"[calculate] 调用 | expression={expression}")
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        logger.info(f"[calculate] 返回 | {result}")
        return str(result)
    except ZeroDivisionError:
        logger.warning("[calculate] 除零错误")
        return "错误：除数不能为零。"
    except Exception as e:
        logger.error(f"[calculate] 计算错误 | {e}")
        return f"计算错误: {e}"


@mcp.tool()
def search_docs(query: str) -> str:
    """
    在项目文档库中执行语义检索（演示版）。

    实际生产环境中，此处会调用 Qdrant / Elasticsearch 等向量数据库进行
    embedding 相似度搜索。本 Demo 仅返回预设片段以展示工具调用链路。

    Args:
        query: 检索关键词，例如 "multi-agent routing"。

    Returns:
        检索结果摘要。
    """
    logger.info(f"[search_docs] 调用 | query={query}")
    mock_fragments = [
        "Multi-Agent 系统通过 Supervisor 动态协调子 Agent，实现任务分发与结果汇总。",
        "IntentClassifier 使用结构化输出（Pydantic）将用户请求映射为预定义意图。",
        "RAG 流程：文档分块 → Embedding → 向量索引 → 相似度检索 → Rerank → 注入上下文。",
        "TeamSupervisorGraph 基于 LangGraph StateGraph，支持可视化、断点续跑与持久化。",
        "MCP（Model Context Protocol）定义了 LLM 与外部工具之间的标准通信协议。",
    ]
    # 简单关键词匹配模拟语义检索
    query_lower = query.lower()
    matched = [frag for frag in mock_fragments if any(kw in frag.lower() for kw in query_lower.split())]
    if not matched:
        matched = mock_fragments[:2]
    result = f"检索 '{query}' 的结果（共 {len(matched)} 条）:\n" + "\n".join(f"- {m}" for m in matched)
    logger.info(f"[search_docs] 返回 | {len(matched)} 条")
    return result


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MCP Demo Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport 类型: stdio（默认，供本地子进程调用）或 sse（HTTP 流）",
    )
    parser.add_argument("--host", default="127.0.0.1", help="SSE 模式监听地址")
    parser.add_argument("--port", type=int, default=18080, help="SSE 模式监听端口")
    args = parser.parse_args()

    if args.transport == "sse":
        logger.info(f"[DemoServer] 启动 SSE 模式 | http://{args.host}:{args.port}/sse")
    else:
        logger.info("[DemoServer] 启动 stdio 模式")

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
