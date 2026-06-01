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
import json
import re
from abc import ABC
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel

from core.agents.toolkit import ToolProvider
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

    组合式工具能力：
    - 通过 tools 参数注入 ToolProvider，Agent 自动进入 ReAct 模式
    - 无 tools 时，Agent 直接调用 LLM
    """

    name: str = "base"
    system_prompt: str = "You are a helpful assistant."
    model_type: ModelType = ModelType.GPT_4O_MINI

    # ReAct 循环最大工具调用轮数
    MAX_TOOL_ROUNDS = 3

    def __init__(
        self,
        model_type: Optional[ModelType] = None,
        llm: Optional[BaseChatModel] = None,
        tools: Optional[List[ToolProvider]] = None,
    ):
        if llm is not None:
            self._llm = llm
            logger.info(f"[{self.name}] 初始化 Agent (外部 LLM 实例)")
        else:
            mt = model_type or self.model_type
            logger.info(f"[{self.name}] 初始化 Agent (model_type={mt.value if hasattr(mt, 'value') else mt})")
            self._llm = llm_factory.get_model(mt)

        self.tools = tools or []

    async def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        执行 Agent 任务。
        - 无 tools -> 直接调用 LLM
        - 有 tools -> ReAct 循环（工具发现 -> LLM 决策 -> 工具执行 -> 结果回传）
        """
        if self.tools:
            return await self._react_run(query, context)
        return await self._direct_run(query, context)

    # ------------------------------------------------------------------
    # 直接运行（无工具）
    # ------------------------------------------------------------------

    async def _direct_run(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """直接调用 LLM，不使用工具"""
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

    # ------------------------------------------------------------------
    # ReAct 运行（有工具）
    # ------------------------------------------------------------------

    async def _react_run(self, query: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        ReAct 主循环。
        使用 AsyncExitStack 统一管理所有 ToolProvider 的生命周期。
        """
        logger.info(f"[{self.name}] 启动 ReAct | query={query[:80]} | providers={len(self.tools)}")

        async with AsyncExitStack() as stack:
            # 1. 进入所有 Provider 的上下文
            for provider in self.tools:
                await stack.enter_async_context(provider)

            # 2. 聚合所有可用工具
            all_tools: List[Any] = []
            for provider in self.tools:
                tool_infos = await provider.discover()
                all_tools.extend(tool_infos)

            tools_desc = self._format_tools(all_tools)
            system_text = (
                f"{self.system_prompt}\n\n"
                f"You have access to the following tools:\n{tools_desc}\n\n"
                "When you need to use a tool, output ONLY a JSON object in this exact format:\n"
                '{"tool": "tool_name", "arguments": {"arg1": "value1"}}\n\n'
                "After receiving the tool result, answer the user in natural language. "
                "If no tool is needed, answer directly."
            )

            messages = [
                SystemMessage(content=system_text),
                HumanMessage(content=query),
            ]

            # 3. ReAct 循环
            for round_num in range(self.MAX_TOOL_ROUNDS):
                logger.info(f"[{self.name}] ReAct round {round_num + 1}/{self.MAX_TOOL_ROUNDS}")
                response = await self._llm.ainvoke(messages)
                content = str(response.content) if response.content else ""

                tool_call = self._extract_tool_call(content)
                if tool_call is None:
                    logger.info(f"[{self.name}] ReAct 完成（无需更多工具）| rounds={round_num + 1}")
                    return content

                # 执行工具调用：遍历所有 provider，第一个成功即返回
                tool_name = tool_call.get("tool")
                tool_args = tool_call.get("arguments", {})
                logger.info(f"[{self.name}] 调用工具 | {tool_name}({tool_args})")

                result_text = ""
                invoked = False
                for provider in self.tools:
                    try:
                        result_text = await provider.invoke(tool_name, tool_args)
                        invoked = True
                        break
                    except Exception as exc:
                        # 当前 provider 无法处理该工具，继续尝试下一个
                        logger.debug(f"[{self.name}] Provider 调用失败，尝试下一个 | {exc}")
                        continue

                if not invoked:
                    result_text = f"Tool execution error: No provider could invoke '{tool_name}'"
                    logger.error(f"[{self.name}] 无可用 Provider 能调用工具 | {tool_name}")

                # 将工具调用与结果追加到对话历史
                messages.append(HumanMessage(content=content))
                messages.append(
                    SystemMessage(
                        content=f"[Tool Result: {tool_name}]\n{result_text}\n"
                        "Now provide the final answer to the user."
                    )
                )

            # 达到最大轮数，强制总结
            logger.warning(f"[{self.name}] ReAct 达到最大轮数，强制总结")
            final_response = await self._llm.ainvoke(messages)
            return str(final_response.content) if final_response.content else ""

    # ------------------------------------------------------------------
    # 工具描述格式化（上提为静态方法）
    # ------------------------------------------------------------------

    @staticmethod
    def _format_tools(tools: List[Any]) -> str:
        """将 Tool 列表格式化为人类可读的文本描述。"""
        lines = []
        for tool in tools:
            name = getattr(tool, "name", "unknown")
            desc = getattr(tool, "description", "No description.")
            schema = getattr(tool, "parameters", {})
            # 简化 schema 展示：只列 properties 的 key
            if isinstance(schema, dict):
                props = schema.get("properties", schema)
                args_desc = ", ".join(props.keys()) if props else "none"
            else:
                args_desc = "none"
            lines.append(f"- {name}({args_desc}): {desc}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Tool-Call 提取（上提为静态方法）
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
        """
        从 LLM 输出中提取 tool-call JSON。

        支持格式:
            1. 纯 JSON 对象（整段文本）
            2. Markdown code block 内的 JSON
        """
        text = text.strip()
        if not text:
            return None

        # 尝试整段解析
        try:
            data = json.loads(text)
            if isinstance(data, dict) and "tool" in data and "arguments" in data:
                return data
        except json.JSONDecodeError:
            pass

        # 尝试从 markdown code block 中提取
        pattern = re.compile(r"```(?:json)?\s*({[\s\S]*?})\s*```", re.IGNORECASE)
        match = pattern.search(text)
        if match:
            try:
                data = json.loads(match.group(1))
                if isinstance(data, dict) and "tool" in data and "arguments" in data:
                    return data
            except json.JSONDecodeError:
                pass

        # 尝试从文本中找最外层 JSON 对象（宽松匹配）
        brace_match = re.search(r"({[\s\S]*?\"tool\"[\s\S]*?})", text)
        if brace_match:
            try:
                data = json.loads(brace_match.group(1))
                if isinstance(data, dict) and "tool" in data and "arguments" in data:
                    return data
            except json.JSONDecodeError:
                pass

        return None

    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name})"
