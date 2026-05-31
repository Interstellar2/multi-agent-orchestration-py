"""
香港法律多 Agent 问答系统 —— 项目主入口

运行方式:
    1. 配置 API Key（.env 文件或环境变量）
    2. 启动 Elasticsearch: docker compose up -d
    3. 构建向量索引: python -m hk_law.rag.ingest --all
    4. 运行 Demo: python main.py

或作为库调用:
    from hk_law.main import HKLawSystem
    system = HKLawSystem()
    result = await system.ask("我被公司无故解雇，可以追讨什么赔偿？", mode="intent")
"""
import asyncio
import sys

from hk_law.main import HKLawSystem, demo
from core.utils.logger import get_logger

logger = get_logger(__name__)


async def interactive_mode():
    """交互式法律问答"""
    system = HKLawSystem()

    logger.info("=" * 60)
    logger.info("香港法律多 Agent 问答系统")
    logger.info("输入问题开始咨询，输入 'quit' 退出")
    logger.info("=" * 60)

    while True:
        try:
            query = input("\n[问题] ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query or query.lower() in ("quit", "exit", "q"):
            break

        result = await system.ask(query, mode="intent")
        logger.info(f"\n[法域] {result['domain']} (置信度: {result['confidence']:.2f})")
        logger.info(f"[回答]\n{result['output']}\n")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="香港法律多 Agent 问答系统")
    parser.add_argument(
        "--mode",
        choices=["demo", "interactive"],
        default="demo",
        help="运行模式: demo (自动跑测试用例) 或 interactive (交互式问答)",
    )
    args = parser.parse_args()

    if args.mode == "interactive":
        await interactive_mode()
    else:
        await demo()


if __name__ == "__main__":
    asyncio.run(main())
