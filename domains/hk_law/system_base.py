"""
DomainSystem —— 多法域多 Agent 系统的可复用基类。

将 HKLawSystem 中的通用路由逻辑（意图识别、Supervisor、语义分析）提取为抽象基类，
新增 Domain 只需实现 agent_factory 和 domain_descriptions 即可。
"""
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional

from core.llm.model_type import ModelType
from core.routing.condition import ConditionRouter
from core.routing.intent import IntentClassifier, LegalQueryAnalyzer
from core.workflows import team_supervisor_graph_workflow
from core.utils.logger import get_logger

logger = get_logger(__name__)


class DomainSystem(ABC):
    """
    多法域多 Agent 系统的抽象基类。

    子类需要实现：
      - _create_agent(domain: str) -> Agent  创建指定法域的 Agent 实例
    """

    def __init__(
        self,
        domains: List[str],
        domain_descriptions: Dict[str, str],
        model_type: ModelType = ModelType.GPT_4O,
    ):
        self.model_type = model_type
        self.domains = domains
        self._domain_descriptions = domain_descriptions
        self._agents: Dict[str, Any] = {}
        logger.info(
            f"[{self.__class__.__name__}] 初始化完成 | model={model_type.value} domains={domains}"
        )

    @abstractmethod
    def _create_agent(self, domain: str):
        """创建指定法域的 Agent 实例。子类必须实现。"""
        ...

    def _get_agent(self, domain: str):
        """懒加载 Agent 实例"""
        if domain not in self._agents:
            self._agents[domain] = self._create_agent(domain)
        return self._agents[domain]

    async def ask_intent_condition(self, query: str) -> Dict:
        """
        模式一：意图识别 + 条件路由。
        先识别法域，再调用对应 Agent。
        """
        logger.info(f"[{self.__class__.__name__}] 模式=intent | query={query[:80]}")
        classifier = IntentClassifier(
            intents=self.domains,
            intent_descriptions=self._domain_descriptions,
            model_type=self.model_type,
        )
        intent_result = await classifier.classify(query)

        default_domain = self.domains[0] if self.domains else None
        router = ConditionRouter.from_intent_map(
            {d: d for d in self.domains},
            default=default_domain,
        )
        target = router.route(intent_result.intent)

        agent = self._get_agent(target)
        output = await agent.run(query)

        logger.info(
            f"[{self.__class__.__name__}] 模式=intent 完成 | domain={target} "
            f"confidence={intent_result.confidence}"
        )
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
        模式二：Team Supervisor（LangGraph 版）。
        Supervisor 动态决定调用哪个法域 Agent，支持多轮（如法域交叉问题）。
        """
        logger.info(f"[{self.__class__.__name__}] 模式=supervisor | query={query[:80]}")
        agent_instances = [self._get_agent(d) for d in self.domains]
        result = await team_supervisor_graph_workflow(
            query,
            agents=agent_instances,
            supervisor_model=self.model_type,
            max_rounds=max_rounds,
        )
        logger.info(
            f"[{self.__class__.__name__}] 模式=supervisor 完成 | "
            f"history={result.get('called_agents')}"
        )
        return result

    async def ask_semantic(self, query: str, max_rounds: int = 2) -> Dict:
        """
        模式三：语义分析 + 智能路由。

        流程：
          1. LegalQueryAnalyzer 解析用户问题（法域、法条、语义改写）
          2. 单法域问题：直接路由到对应 Agent，使用改写后的查询做 RAG
          3. 跨法域问题：仅调度涉及的法域 Agent，通过 Supervisor 协作
        """
        logger.info(f"[{self.__class__.__name__}] 模式=semantic | query={query[:80]}")

        analyzer = LegalQueryAnalyzer(
            domains=self.domains,
            domain_descriptions=self._domain_descriptions,
            model_type=self.model_type,
        )
        analysis = await analyzer.analyze(query)

        # 单法域：直接路由（最优路径）
        if not analysis.is_cross_domain and len(analysis.jurisdictions) == 1:
            target = analysis.jurisdictions[0]
            agent = self._get_agent(target)
            output = await agent.run(
                query,
                rewritten_query=analysis.rewritten_query,
                statutes=analysis.statutes,
            )
            logger.info(
                f"[{self.__class__.__name__}] 模式=semantic 完成 | routing=direct "
                f"domain={target} intent={analysis.intent} confidence={analysis.confidence}"
            )
            return {
                "mode": "semantic",
                "routing": "direct",
                "intent": analysis.intent,
                "jurisdictions": analysis.jurisdictions,
                "statutes": analysis.statutes,
                "rewritten_query": analysis.rewritten_query,
                "is_cross_domain": analysis.is_cross_domain,
                "confidence": analysis.confidence,
                "reason": analysis.reason,
                "domain": target,
                "output": output,
            }

        # 跨法域：走 Supervisor，仅调度涉及法域
        involved_domains = [d for d in analysis.jurisdictions if d in self.domains]
        if not involved_domains:
            involved_domains = self.domains[:1]
            analysis.is_cross_domain = False

        involved_agents = [self._get_agent(d) for d in involved_domains]

        enhanced_query = (
            f"[语义分析] 该问题涉及法域：{', '.join(analysis.jurisdictions)}；"
            f"涉及法条：{', '.join(analysis.statutes) if analysis.statutes else '未明确'}；"
            f"核心意图：{analysis.intent}。\n\n"
            f"[原始问题] {query}"
        )

        result = await team_supervisor_graph_workflow(
            enhanced_query,
            agents=involved_agents,
            supervisor_model=self.model_type,
            max_rounds=max_rounds,
        )
        logger.info(
            f"[{self.__class__.__name__}] 模式=semantic 完成 | routing=supervisor "
            f"domains={involved_domains} history={result.get('called_agents')}"
        )
        result["mode"] = "semantic"
        result["routing"] = "supervisor"
        result["semantic_analysis"] = {
            "intent": analysis.intent,
            "jurisdictions": analysis.jurisdictions,
            "statutes": analysis.statutes,
            "rewritten_query": analysis.rewritten_query,
            "is_cross_domain": analysis.is_cross_domain,
            "confidence": analysis.confidence,
            "reason": analysis.reason,
        }
        return result

    async def ask(self, query: str, mode: str = "intent") -> Dict:
        """
        统一入口。

        Args:
            query: 用户问题
            mode: "intent" / "supervisor" / "semantic"
        """
        logger.info(f"[{self.__class__.__name__}] ask | mode={mode} query={query[:80]}")
        if mode == "intent":
            return await self.ask_intent_condition(query)
        elif mode == "supervisor":
            return await self.ask_supervisor(query)
        elif mode == "semantic":
            return await self.ask_semantic(query)
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Use 'intent', 'supervisor' or 'semantic'."
            )
