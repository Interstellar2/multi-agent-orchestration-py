"""
Agent 基类
所有子 Agent 继承此类，实现 run 方法即可被工作流调用。

LLM 配置（三选一，优先级从高到低）：
  1. llm: 直接传入 BaseChatModel 实例（最灵活，任意提供商）
  2. model_type: 通过 ModelType 枚举从工厂获取（推荐，统一管理）
  3. 类属性 self.model_type: 默认模型

示例:
    from llm.model_type import ModelType
    from llm.factory import llm_factory

    # 方式一：指定模型类型（通过工厂统一管理 API Key）
    agent = CodeAgent(model_type=ModelType.QWEN_MAX)

    # 方式二：直接传入 LLM 实例
    from langchain_anthropic import ChatAnthropic
    agent = CodeAgent(llm=ChatAnthropic(model="claude-3-5-sonnet"))

    # 方式三：用默认模型
    agent = CodeAgent()
"""
from abc import ABC
from typing import Any, Dict, Optional

from langchain.schema import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from core.llm.factory import llm_factory
from core.llm.model_type import ModelType
from core.utils.logger import get_logger

logger = get_logger(__name__)


class Agent(ABC):
    """
    Agent 基类。子类只需：
    1. 设置 name 和 system_prompt
    2. （可选）设置默认 model_type
    3. （可选）重写 run 方法
    """

    name: str = "base"
    system_prompt: str = "You are a helpful assistant."
    model_type: ModelType = ModelType.GPT_4O_MINI

    def __init__(
        self,
        model_type: Optional[ModelType] = None,
        llm: Optional[BaseChatModel] = None,
    ):
        if llm is not None:
            self._llm = llm
            logger.info(f"[{self.name}] 初始化 Agent (外部 LLM 实例)")
        else:
            mt = model_type or self.model_type
            logger.info(f"[{self.name}] 初始化 Agent (model_type={mt.value if hasattr(mt, 'value') else mt})")
            self._llm = llm_factory.get_model(mt)

    async def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        执行 Agent 任务。
        子类可以直接继承此默认实现，也可以重写。
        """
        logger.info(f"[{self.name}] 开始运行 | query={query[:80]}")
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=query),
        ]
        try:
            response = await self._llm.ainvoke(messages)
            output = response.content
            logger.info(f"[{self.name}] 运行完成 | output_len={len(output)}")
            return output
        except Exception as e:
            logger.error(f"[{self.name}] 运行失败 | error={e}")
            raise

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"
