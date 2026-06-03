"""
香港法律多 Agent 系统 — 业务入口

功能：
  1. 根据用户问题识别法律法域（意图识别）
  2. 调度对应法域的 Agent 进行回答
  3. 每个 Agent 自带 RAG，基于实际法律条文回答

运行方式:
    python -m domains.hk_law.main

或:
    from domains.hk_law.main import HKLawSystem
    system = HKLawSystem()
    result = await system.ask("我被公司无故解雇，可以追讨什么赔偿？")
"""
import asyncio

from typing import Dict

from domains.hk_law.agents import get_hk_law_agent, list_domains
from domains.hk_law.agents import _DOMAIN_CONFIGS
from domains.hk_law.system_base import DomainSystem
from core.llm.model_type import ModelType
from core.utils.logger import get_logger

logger = get_logger(__name__)


class HKLawSystem(DomainSystem):
    """
    香港法律多 Agent 系统。
    继承 DomainSystem 基类，只保留 domain-specific 配置。
    """

    def __init__(self, model_type: ModelType = ModelType.GPT_4O):
        domains = list_domains()
        domain_descriptions = {
            "criminal": "刑事案件、犯罪指控、刑事程序、警方调查、保释、量刑",
            "civil": "民事纠纷、合约争议、侵权申索、债务追讨、民事诉讼",
            "company": "公司注册、董事责任、股东权益、公司清盘、上市公司合规",
            "employment": "劳动合同、解雇、遣散费、歧视、工伤、劳资纠纷",
            "property": "楼宇买卖、租赁、公契、管理费、业主立案法团",
        }
        super().__init__(
            domains=domains,
            domain_descriptions=domain_descriptions,
            model_type=model_type,
        )

    def _create_agent(self, domain: str):
        """创建指定法域的 HKLawAgent 实例。"""
        return get_hk_law_agent(domain, model_type=self.model_type)


async def demo():
    """演示：香港法律多 Agent 系统（含语义分析模式）"""
    system = HKLawSystem(model_type=ModelType.GPT_4O_MINI)

    test_cases = [
        # 原有模式对比
        ("intent", "我在公司工作了3年，今天被老板突然解雇，没有给任何通知，我可以追讨什么赔偿？"),
        ("supervisor", "我被公司解雇，同时公司还欠我两个月工资，我该怎么办？"),
        # 语义分析模式（单法域）
        ("semantic", "我在公司工作了3年，今天被老板突然解雇，没有给任何通知，我可以追讨什么赔偿？"),
        # 语义分析模式（可能跨法域：噪音 + 刑事）
        ("semantic", "我的邻居经常在半夜制造噪音，我可以报警吗？这属于刑事罪行吗？"),
        # 语义分析模式（跨法域：公司 + 雇佣）
        ("semantic", "我被公司解雇，同时公司还欠我两个月工资，我该怎么办？"),
    ]

    for mode, query in test_cases:
        logger.info("=" * 70)
        logger.info(f"[模式: {mode}] [问题] {query}")
        logger.info("=" * 70)

        result = await system.ask(query, mode=mode)

        if mode == "intent":
            logger.info(f"识别法域: {result['domain']} (置信度: {result['confidence']}) | 理由: {result['reason']}")

        if mode == "semantic":
            if result.get("routing") == "direct":
                logger.info(
                    f"[语义分析] 法域: {result['jurisdictions']} | 意图: {result['intent']} | "
                    f"跨域: {result['is_cross_domain']} | 置信度: {result['confidence']}"
                )
                logger.info(f"[改写查询] {result['rewritten_query']}")
                if result.get("statutes"):
                    logger.info(f"[涉及法条] {result['statutes']}")
                logger.info(f"[路由策略] 单法域直达 -> {result['domain']}")
            else:
                sa = result.get("semantic_analysis", {})
                logger.info(
                    f"[语义分析] 法域: {sa.get('jurisdictions')} | 意图: {sa.get('intent')} | "
                    f"跨域: {sa.get('is_cross_domain')}"
                )
                logger.info(f"[路由策略] Supervisor 跨域协作 | 历史: {result.get('called_agents')}")

        logger.info(f"[回答] {result['output']}")

        if mode in ("supervisor", "semantic") and result.get("history"):
            history_str = [f"Round {s['round']}: {s['agent']}" for s in result['history']]
            logger.info(f"调用历史: {history_str}")


if __name__ == "__main__":
    asyncio.run(demo())
