"""
条件路由模块
基于规则匹配做分支判断，不需要 LLM，速度快、确定性高。
"""
from typing import Any, Callable, Dict, Optional


class ConditionRouter:
    """
    条件路由器。
    用法：
        router = ConditionRouter()
        router.add_rule("code", lambda x: x == "code")
        target = router.route("code")  # -> "code"
    """

    def __init__(self):
        self.rules: Dict[str, Callable[[Any], bool]] = {}
        self.default: Optional[str] = None

    def add_rule(self, name: str, matcher: Callable[[Any], bool]):
        self.rules[name] = matcher
        return self

    def set_default(self, name: str):
        self.default = name
        return self

    def route(self, value: Any) -> Optional[str]:
        for name, matcher in self.rules.items():
            if matcher(value):
                return name
        return self.default

    @classmethod
    def from_intent_map(cls, intent_map: Dict[str, str], default: str = None):
        """
        快捷创建：intent -> agent_name 的映射。
        例如 {"code": "code", "search": "search", "chat": "chat"}
        """
        router = cls()
        for intent, agent_name in intent_map.items():
            router.add_rule(agent_name, lambda v, i=intent: v == i)
        if default:
            router.set_default(default)
        return router
