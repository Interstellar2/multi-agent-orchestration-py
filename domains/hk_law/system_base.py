"""
DomainSystem —— 多法域多 Agent 系统的可复用基类（支持多轮对话）。

将 HKLawSystem 中的通用路由逻辑（意图识别、Supervisor、语义分析）提取为抽象基类，
新增 Domain 只需实现 agent_factory 和 domain_descriptions 即可。

多轮对话能力：
  - 通过 ConversationManager 管理历史窗口和持久化
  - Fast-Path 法域锁定优化连续追问
  - 支持 /new 切换会话
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from core.llm.model_type import ModelType
from core.routing.condition import ConditionRouter
from core.routing.intent import IntentClassifier, LegalQueryAnalyzer
from core.session import ConversationManager, SessionStore, InMemorySessionStore
from core.workflows import team_supervisor_graph_workflow
from core.utils.logger import get_logger

logger = get_logger(__name__)


class DomainSystem(ABC):
    """
    多法域多 Agent 系统的抽象基类（多轮对话版）。

    子类需要实现：
      - _create_agent(domain: str) -> Agent  创建指定法域的 Agent 实例
    """

    def __init__(
        self,
        domains: List[str],
        domain_descriptions: Dict[str, str],
        model_type: ModelType = ModelType.GPT_4O,
        session_store: Optional[SessionStore] = None,
        event_callback=None,
    ):
        self.model_type = model_type
        self.domains = domains
        self._domain_descriptions = domain_descriptions
        self._agents: Dict[str, Any] = {}
        self._event_callback = event_callback
        self.session_manager = ConversationManager(
            store=session_store or InMemorySessionStore(),
            max_turns=10,
        )
        logger.info(
            f"[{self.__class__.__name__}] 初始化完成 | model={model_type.value} domains={domains}"
        )

    @abstractmethod
    def _create_agent(self, domain: str):
        """创建指定法域的 Agent 实例。子类必须实现。"""
        ...

    def _get_agent(self, domain: str):
        """懒加载 Agent 实例，并注入当前 event_callback。"""
        if domain not in self._agents:
            self._agents[domain] = self._create_agent(domain)
        # 确保已存在的 agent 也能收到最新的 callback（支持 API 层复用）
        if hasattr(self._agents[domain], "event_callback"):
            self._agents[domain].event_callback = self._event_callback
        return self._agents[domain]

    @staticmethod
    def _history_to_text(history: Optional[List[BaseMessage]]) -> str:
        """将 BaseMessage 列表转换为意图识别器可读的文本。"""
        if not history:
            return ""
        lines = []
        for msg in history:
            if isinstance(msg, HumanMessage):
                lines.append(f"用户: {msg.content[:200]}")
            elif isinstance(msg, AIMessage):
                lines.append(f"助手: {msg.content[:200]}")
        return "\n".join(lines[-6:])  # 只取最近 3 轮 human+ai

    # ------------------------------------------------------------------
    # 三种模式（均支持多轮历史）
    # ------------------------------------------------------------------

    async def ask_intent_condition(
        self,
        query: str,
        history: Optional[List[BaseMessage]] = None,
        session_id: Optional[str] = None,
    ) -> Dict:
        """
        模式一：意图识别 + 条件路由。
        先识别法域，再调用对应 Agent。
        """
        logger.info(f"[{self.__class__.__name__}] 模式=intent | query={query[:80]}")

        # 1. 意图识别（注入历史文本做上下文感知）
        classifier = IntentClassifier(
            intents=self.domains,
            intent_descriptions=self._domain_descriptions,
            model_type=self.model_type,
        )
        history_text = self._history_to_text(history)
        intent_result = await classifier.classify(query, history=history_text)
        await self._emit("intent", {
            "domain": intent_result.intent,
            "confidence": intent_result.confidence,
            "reason": intent_result.reason,
        })

        default_domain = self.domains[0] if self.domains else None
        router = ConditionRouter.from_intent_map(
            {d: d for d in self.domains},
            default=default_domain,
        )
        target = router.route(intent_result.intent)

        # 2. 执行 Agent（注入多轮历史）
        agent = self._get_agent(target)
        output = await agent.run(query, history=history)

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

    async def ask_supervisor(
        self,
        query: str,
        history: Optional[List[BaseMessage]] = None,
        max_rounds: int = 2,
    ) -> Dict:
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
            history=history,
            event_callback=self._event_callback,
        )
        logger.info(
            f"[{self.__class__.__name__}] 模式=supervisor 完成 | "
            f"history={result.get('called_agents')}"
        )
        return result

    async def ask_semantic(
        self,
        query: str,
        history: Optional[List[BaseMessage]] = None,
        max_rounds: int = 2,
    ) -> Dict:
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
        history_text = self._history_to_text(history)
        analysis = await analyzer.analyze(query, history=history_text)
        await self._emit("semantic", {
            "intent": analysis.intent,
            "jurisdictions": analysis.jurisdictions,
            "statutes": analysis.statutes,
            "rewritten_query": analysis.rewritten_query,
            "is_cross_domain": analysis.is_cross_domain,
            "confidence": analysis.confidence,
            "reason": analysis.reason,
        })

        # 单法域：直接路由（最优路径）
        if not analysis.is_cross_domain and len(analysis.jurisdictions) == 1:
            target = analysis.jurisdictions[0]
            agent = self._get_agent(target)
            output = await agent.run(
                query,
                history=history,
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
            history=history,
            event_callback=self._event_callback,
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

    # ------------------------------------------------------------------
    # 统一入口（多轮对话核心逻辑）
    # ------------------------------------------------------------------

    async def _emit(self, event_type: str, data: Dict[str, Any]) -> None:
        """如果设置了 event_callback，则推送事件。"""
        if self._event_callback is not None:
            try:
                await self._event_callback({"type": event_type, "data": data})
            except Exception as exc:
                logger.debug(f"[{self.__class__.__name__}] event_callback 异常（忽略）: {exc}")

    async def ask(
        self,
        query: str,
        mode: str = "intent",
        session_id: Optional[str] = None,
        event_callback=None,
    ) -> Dict:
        """
        统一入口（支持多轮对话）。

        Args:
            query: 用户问题
            mode: "intent" / "supervisor" / "semantic"
            session_id: 会话 ID（None 表示单轮 Stateless）
            event_callback: 可选的事件回调（用于 SSE 推送）
        """
        if event_callback is not None:
            self._event_callback = event_callback

        logger.info(
            f"[{self.__class__.__name__}] ask | mode={mode} "
            f"session={session_id or 'N/A'} query={query[:80]}"
        )
        await self._emit("start", {"mode": mode, "query": query[:200], "session_id": session_id})

        # 1. 加载历史（如有 session_id）
        history = []
        if session_id:
            history = await self.session_manager.load_history(session_id)

        # 2. Fast-Path 法域锁定（仅 intent/semantic 模式）
        fast_domain = None
        if session_id and mode in ("intent", "semantic"):
            fast_domain = await self.session_manager.check_fast_path_async(
                session_id, query
            )

        # 3. 执行当前轮
        if fast_domain and fast_domain in self.domains:
            # Fast-Path：跳过意图识别，直接命中法域
            logger.info(f"[{self.__class__.__name__}] FastPath 命中 | domain={fast_domain}")
            await self._emit("fastpath", {"domain": fast_domain})
            agent = self._get_agent(fast_domain)
            output = await agent.run(query, history=history)
            result = {
                "mode": f"{mode}_fastpath",
                "domain": fast_domain,
                "output": output,
                "fastpath": True,
            }
        elif mode == "intent":
            result = await self.ask_intent_condition(
                query, history=history, session_id=session_id
            )
        elif mode == "supervisor":
            result = await self.ask_supervisor(query, history=history)
        elif mode == "semantic":
            result = await self.ask_semantic(query, history=history)
        else:
            raise ValueError(
                f"Unknown mode: {mode}. Use 'intent', 'supervisor' or 'semantic'."
            )

        await self._emit("done", {"mode": mode, "domain": result.get("domain"), "output": result["output"][:500]})

        # 4. 保存当前轮到 Session
        if session_id:
            domain = result.get("domain", result.get("semantic_analysis", {}).get("jurisdictions", ["unknown"])[0])
            confidence = result.get("confidence", result.get("semantic_analysis", {}).get("confidence", 0.0))
            await self.session_manager.append_turn(
                session_id, "human", query, metadata={"mode": mode}
            )
            await self.session_manager.append_turn(
                session_id,
                "ai",
                result["output"],
                metadata={
                    "domain": domain,
                    "mode": mode,
                    "confidence": confidence,
                    "fastpath": result.get("fastpath", False),
                },
            )

        return result
