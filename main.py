"""
香港法律多 Agent 问答系统 —— 项目主入口

运行方式:
    1. 配置 API Key（.env 文件或环境变量）
    2. 启动 Elasticsearch: docker compose up -d
    3. 构建向量索引: python -m domains.hk_law.rag.ingest --all
    4. 运行 Demo: python main.py

或作为库调用:
    from domains.hk_law.main import HKLawSystem
    system = HKLawSystem()
    result = await system.ask("我被公司无故解雇，可以追讨什么赔偿？", mode="intent")
"""
import asyncio
import sys

from core.utils.logger import get_logger
from core.workflows import (
    mcp_react_workflow,
    mcp_supervisor_workflow,
)
from domains.capital_market.main import demo as capital_market_demo

logger = get_logger(__name__)


async def interactive_mode():
    """交互式法律问答（支持多轮对话）"""
    import uuid
    from core.session import PostgresSessionStore
    from domains.hk_law.main import HKLawSystem

    # 优先使用 PostgreSQL，连接失败则回退到内存
    try:
        store = PostgresSessionStore()
        system = HKLawSystem(session_store=store)
        logger.info("[Session] 使用 PostgreSQL 存储")
    except Exception as e:
        logger.warning(f"[Session] PostgreSQL 连接失败，回退到内存存储 | {e}")
        system = HKLawSystem()

    session_id = str(uuid.uuid4())
    logger.info("=" * 60)
    logger.info("香港法律多 Agent 问答系统")
    logger.info("输入问题开始咨询，输入 'quit' 退出，'/new' 开启新对话")
    logger.info(f"当前会话 ID: {session_id}")
    logger.info("=" * 60)

    while True:
        try:
            query = input("\n[问题] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            break

        if query == "/new":
            session_id = str(uuid.uuid4())
            logger.info(f"[会话] 已开启新对话 | session_id={session_id}")
            continue

        result = await system.ask(query, mode="intent", session_id=session_id)
        confidence = result.get("confidence", 0.0)
        domain = result.get("domain", "unknown")
        logger.info(f"\n[法域] {domain} (置信度: {confidence:.2f})")
        if result.get("fastpath"):
            logger.info("[路由] Fast-Path 法域锁定")
        logger.info(f"[回答]\n{result['output']}\n")


async def mcp_demo(mode: str = "react", transport: str = "stdio", server_url: str = None):
    """
    MCP 工具调用演示入口。

    参数:
        mode      : "react" 单独跑 MCPAgent，"supervisor" 跑 Supervisor+MCPAgent
        transport : "stdio" 或 "sse"
        server_url: SSE 模式下的服务器地址（stdio 模式下忽略）
    """
    logger.info("=" * 60)
    logger.info("MCP 工具调用演示")
    logger.info("=" * 60)

    if transport == "sse" and not server_url:
        server_url = "http://127.0.0.1:18080/sse"

    test_queries = [
        "北京今天天气怎么样？",
        "请帮我计算 (128 + 256) * 4 等于多少？",
        "搜索一下 multi-agent routing 的相关文档",
        "香港的天气如何，顺便算一下 100 除以 4？",
    ]

    for query in test_queries:
        logger.info(f"\n[测试问题] {query}")
        try:
            if mode == "react":
                result = await mcp_react_workflow(
                    query=query,
                    server_url=server_url if transport == "sse" else None,
                    server_cmd=None if transport == "sse" else [
                        "python", "-m", "mcp_bridge.server.demo_server", "--transport", "stdio"
                    ],
                )
            else:
                result = await mcp_supervisor_workflow(
                    query=query,
                    server_url=server_url if transport == "sse" else None,
                    server_cmd=None if transport == "sse" else [
                        "python", "-m", "mcp_bridge.server.demo_server", "--transport", "stdio"
                    ],
                )
            logger.info(f"[结果] {result.get('output', result.get('final_output', 'N/A'))}\n")
        except Exception as e:
            logger.error(f"[错误] {e}")
            import traceback
            traceback.print_exc()

    logger.info("=" * 60)
    logger.info("MCP 演示完成")
    logger.info("=" * 60)


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Multi-Agent Demo 主入口")
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # 香港法律子命令
    law_parser = subparsers.add_parser("hk_law", help="香港法律多 Agent 问答系统")
    law_parser.add_argument(
        "--mode",
        choices=["demo", "interactive"],
        default="demo",
        help="运行模式: demo (自动跑测试用例) 或 interactive (交互式问答)",
    )

    # MCP 演示子命令
    mcp_parser = subparsers.add_parser("mcp", help="MCP 工具调用演示")
    mcp_parser.add_argument(
        "--mode",
        choices=["react", "supervisor"],
        default="react",
        help="MCP 演示模式: react (单独 MCPAgent) 或 supervisor (Supervisor+MCPAgent)",
    )
    mcp_parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP Server transport: stdio (默认) 或 sse",
    )
    mcp_parser.add_argument(
        "--server-url",
        default=None,
        help="SSE 模式下 MCP Server 地址，例如 http://127.0.0.1:18080/sse",
    )

    # 资本市场研究助理子命令
    cm_parser = subparsers.add_parser(
        "capital_market", help="香港资本市场研究助理（连接 hk-finance-mcp）"
    )
    cm_parser.add_argument(
        "--mode",
        choices=["research", "team"],
        default="research",
        help="运行模式: research (单独 CapitalMarketAgent) 或 team (Supervisor 协作)",
    )
    cm_parser.add_argument(
        "--server-url",
        default=None,
        help="hk-finance-mcp SSE 地址，默认 http://127.0.0.1:1888/mcp/sse",
    )

    args = parser.parse_args()

    if args.command == "mcp":
        await mcp_demo(mode=args.mode, transport=args.transport, server_url=args.server_url)
    elif args.command == "capital_market":
        await capital_market_demo(mode=args.mode, server_url=args.server_url)
    elif args.command == "hk_law":
        if args.mode == "interactive":
            await interactive_mode()
        else:
            from domains.hk_law.main import demo
            await demo()
    else:
        # 默认行为：向后兼容，跑 hk_law demo
        from domains.hk_law.main import demo
        await demo()


if __name__ == "__main__":
    asyncio.run(main())
