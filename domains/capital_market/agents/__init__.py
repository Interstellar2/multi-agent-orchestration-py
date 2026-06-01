"""
资本市场 Agent

用法:
    from domains.capital_market.agents import CapitalMarketAgent

    agent = CapitalMarketAgent()
    output = await agent.run("查询腾讯控股最近一年的回购记录")
"""
from domains.capital_market.agents.base import CapitalMarketAgent

__all__ = ["CapitalMarketAgent"]
