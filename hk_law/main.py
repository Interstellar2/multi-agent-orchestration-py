"""
香港法律多 Agent 系统 — 业务入口

功能：
  1. 根据用户问题识别法律法域（意图识别）
  2. 调度对应法域的 Agent 进行回答
  3. 每个 Agent 自带 RAG，基于实际法律条文回答

运行方式:
    python -m hk_law.main

或:
    from hk_law.main import HKLawSystem
    system = HKLawSystem()
    result = await system.ask("我被公司无故解雇，可以追讨什么赔偿？")
"""
import asyncio

from typing import Dict

from hk_law.agents import get_hk_law_agent, list_domains
from hk_law.agents.base import HKLawAgent
from core.llm.model_type import ModelType
from core.routing.condition import ConditionRouter
from core.routing.intent import IntentClassifier
from core.workflows import team_supervisor_graph_workflow
from core.utils.logger import get_logger

logger = get_logger(__name__)


class HKLawSystem:
    """
    香港法律多 Agent 系统。
    封装了意图识别、Agent 调度和 RAG 检索的完整流程。
    """

    def __init__(self, model_type: ModelType = ModelType.GPT_4O):
        self.model_type = model_type
        self.domains = list_domains()
        self._agents: Dict[str, HKLawAgent] = {}
        logger.info(f"[HKLawSystem] 初始化完成 | model={model_type.value} domains={self.domains}")

    def _get_agent(self, domain: str) -> HKLawAgent:
        """懒加载 Agent 实例"""
        if domain not in self._agents:
            self._agents[domain] = get_hk_law_agent(domain, model_type=self.model_type)
        return self._agents[domain]

    async def ask_intent_condition(self, query: str) -> Dict:
        """
        模式一：意图识别 + 条件路由
        先识别法域，再调用对应 Agent。
        """
        logger.info(f"[HKLawSystem] 模式=intent | query={query[:80]}")
        # 意图识别
        classifier = IntentClassifier(
            intents=self.domains,
            intent_descriptions={
                "criminal": "刑事案件、犯罪指控、刑事程序、警方调查、保释、量刑",
                "civil": "民事纠纷、合约争议、侵权申索、债务追讨、民事诉讼",
                "company": "公司注册、董事责任、股东权益、公司清盘、上市公司合规",
                "employment": "劳动合同、解雇、遣散费、歧视、工伤、劳资纠纷",
                "property": "楼宇买卖、租赁、公契、管理费、业主立案法团",
            },
            model_type=self.model_type,
        )
        intent_result = await classifier.classify(query)

        # 条件路由
        default_domain = self.domains[0] if self.domains else None
        router = ConditionRouter.from_intent_map(
            {d: d for d in self.domains},
            default=default_domain,
        )
        target = router.route(intent_result.intent)

        # 执行 Agent
        agent = self._get_agent(target)
        output = await agent.run(query)

        logger.info(f"[HKLawSystem] 模式=intent 完成 | domain={target} confidence={intent_result.confidence}")
        return {
            "mode": "intent_condition",
            "intent": intent_result.intent,
            "confidence": intent_result.confidence,
            "reason": intent_result.reason,
            "domain": target,
            "output": output,
        }

    async def ask_supervisor(self, query: str, max_rounds: int = 2) -> Dict:
        """
        模式二：Team Supervisor（LangGraph 版）
        Supervisor 动态决定调用哪个法域 Agent，支持多轮（如法域交叉问题）。
        """
        logger.info(f"[HKLawSystem] 模式=supervisor | query={query[:80]}")
        agent_instances = [self._get_agent(d) for d in self.domains]
        result = await team_supervisor_graph_workflow(
            query,
            agents=agent_instances,
            supervisor_model=self.model_type,
            max_rounds=max_rounds,
        )
        logger.info(f"[HKLawSystem] 模式=supervisor 完成 | history={result.get('called_agents')}")
        return result

    async def ask(self, query: str, mode: str = "intent") -> Dict:
        """
        统一入口。

        Args:
            query: 用户的法律问题
            mode: "intent" (意图识别+条件路由) 或 "supervisor" (动态协调)
        """
        logger.info(f"[HKLawSystem] ask | mode={mode} query={query[:80]}")
        if mode == "intent":
            return await self.ask_intent_condition(query)
        elif mode == "supervisor":
            return await self.ask_supervisor(query)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'intent' or 'supervisor'.")


async def demo():
    """演示：香港法律多 Agent 系统"""
    system = HKLawSystem(model_type=ModelType.GPT_4O_MINI)

    test_cases = [
        ("intent", "我在公司工作了3年，今天被老板突然解雇，没有给任何通知，我可以追讨什么赔偿？"),
        ("intent", "我的邻居经常在半夜制造噪音，我可以报警吗？这属于刑事罪行吗？"),
        ("intent", "我想在香港注册一家有限公司，需要什么文件？董事有什么法律责任？"),
        ("intent", "我租的公寓漏水，房东不肯维修，我可以扣起租金吗？"),
        ("supervisor", "我被公司解雇，同时公司还欠我两个月工资，我该怎么办？"),
    ]

    for mode, query in test_cases:
        logger.info("=" * 70)
        logger.info(f"[模式: {mode}] [问题] {query}")
        logger.info("=" * 70)

        result = await system.ask(query, mode=mode)

        if mode == "intent":
            logger.info(f"识别法域: {result['domain']} (置信度: {result['confidence']}) | 理由: {result['reason']}")

        logger.info(f"[回答] {result['output']}")

        if mode == "supervisor" and result.get("history"):
            history_str = [f"Round {s['round']}: {s['agent']}" for s in result['history']]
            logger.info(f"调用历史: {history_str}")


if __name__ == "__main__":
    asyncio.run(demo())
