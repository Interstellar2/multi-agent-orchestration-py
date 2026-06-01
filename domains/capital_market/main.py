"""
香港资本市场研究助理 — 业务入口

功能：
  1. 直连 hk-finance-mcp 查询港交所金融数据
  2. 支持独立运行（单 Agent ReAct）和团队模式（Supervisor + ChatAgent）

运行方式:
    python -m domains.capital_market.main

或:
    from domains.capital_market.main import capital_market_research_workflow
    result = await capital_market_research_workflow("查询腾讯控股最近一年的回购记录")
"""
from typing import Dict, List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from core.agents.base import Agent
from core.llm.model_type import ModelType
from core.routing.supervisor import TeamSupervisor
from core.workflows import _resolve_agents
from domains.capital_market.agents import CapitalMarketAgent
from core.utils.logger import get_logger

logger = get_logger(__name__)


async def capital_market_research_workflow(
    query: str,
    server_url: Optional[str] = None,
    server_cmd: Optional[List[str]] = None,
    model_type: ModelType = None,
    llm: Optional[BaseChatModel] = None,
) -> Dict:
    """
    资本市场研究助理（独立模式）

    专门处理港交所金融数据查询，通过 MCP 协议调用 text2sql、
    向量检索、公司名称模糊匹配等真实业务工具。
    """
    logger.info(f"[Workflow] 启动资本市场研究助理 | query={query[:80]}")

    agent = CapitalMarketAgent(
        model_type=model_type,
        llm=llm,
        server_url=server_url,
        server_cmd=server_cmd,
    )
    output = await agent.run(query)
    logger.info(f"[Workflow] 完成资本市场研究助理 | output_len={len(output)}")
    return {
        "mode": "capital_market_research",
        "agent": agent.name,
        "output": output,
    }


async def capital_market_team_workflow(
    query: str,
    server_url: Optional[str] = None,
    server_cmd: Optional[List[str]] = None,
    model_type: ModelType = None,
    llm: Optional[BaseChatModel] = None,
    supervisor_model: ModelType = None,
    supervisor_llm: Optional[BaseChatModel] = None,
    max_rounds: int = 3,
) -> Dict:
    """
    Team Supervisor + 资本市场研究助理（团队协作模式）

    在 Supervisor 团队中注入 CapitalMarketAgent 作为金融数据专家。
    用户问港股/金融问题时路由给它，问一般问题时路由给 ChatAgent。
    """
    logger.info(f"[Workflow] 启动资本市场团队协作 | query={query[:80]}")

    capital_agent = CapitalMarketAgent(
        model_type=model_type,
        llm=llm,
        server_url=server_url,
        server_cmd=server_cmd,
    )
    base_agents = _resolve_agents(
        ["chat"],
        model_type=model_type,
        llm=llm,
    )
    all_agents = base_agents + [capital_agent]  # type: ignore[list-item]

    supervisor = TeamSupervisor(
        agents=all_agents,
        model_type=supervisor_model or model_type,
        llm=supervisor_llm,
        max_rounds=max_rounds,
    )
    result = await supervisor.run(query)
    logger.info(
        f"[Workflow] 完成资本市场团队协作 | "
        f"调用 history={result.get('called_agents')}"
    )
    return {"mode": "capital_market_team", **result}


async def demo(
    mode: str = "research",
    server_url: str = None,
):
    """
    资本市场研究助理演示入口。

    参数:
        mode: "research" 单独跑 CapitalMarketAgent，
              "team" 跑 Supervisor + CapitalMarketAgent + ChatAgent
        server_url: hk-finance-mcp SSE 地址
    """
    logger.info("=" * 60)
    logger.info("香港资本市场研究助理演示")
    logger.info("=" * 60)

    if not server_url:
        server_url = "http://127.0.0.1:1888/mcp/sse"
        logger.info(f"使用默认 MCP Server: {server_url}")
        logger.info(
            "请确保 hk-finance-mcp 已启动: "
            "cd /Users/hanniandong/python_project/hk-finance-mcp && python main.py"
        )

    test_queries = [
        "查询小米集团的股票代码和上市信息",
        "最近一年有哪些公司在港交所回购股票？",
        "中金的保荐项目有哪些？",
        "查询'腾讯'的官方公司全称",
    ]

    for query in test_queries:
        logger.info(f"\n[测试问题] {query}")
        try:
            if mode == "research":
                result = await capital_market_research_workflow(
                    query=query,
                    server_url=server_url,
                )
                output = result.get("output", "N/A")
            else:
                result = await capital_market_team_workflow(
                    query=query,
                    server_url=server_url,
                )
                output = result.get("final_output", "N/A")
            logger.info(f"[结果]\n{output}\n")
        except Exception as e:
            logger.error(f"[错误] {e}")
            import traceback
            traceback.print_exc()

    logger.info("=" * 60)
    logger.info("演示完成")
    logger.info("=" * 60)


if __name__ == "__main__":
    import asyncio
    import argparse

    parser = argparse.ArgumentParser(description="香港资本市场研究助理")
    parser.add_argument(
        "--mode",
        choices=["research", "team"],
        default="research",
        help="运行模式: research (单独 Agent) 或 team (Supervisor 协作)",
    )
    parser.add_argument(
        "--server-url",
        default=None,
        help="hk-finance-mcp SSE 地址，默认 http://127.0.0.1:1888/mcp/sse",
    )
    args = parser.parse_args()
    asyncio.run(demo(mode=args.mode, server_url=args.server_url))
