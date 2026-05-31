"""
Multi-Agent + Intent Recognition 框架示例

演示 core/ 框架的 6 种使用模式：
  1. 意图识别 + 条件路由
  2. Team Supervisor（Python 循环版）
  3. Team Supervisor（LangGraph 版）
  4. 混合模型（每个 Agent 不同 LLM）
  5. 工厂 API 与动态注册
  6. 扩展自定义 Agent

运行方式:
    python -m core.demo
"""
import asyncio

from core.workflows import (
    intent_condition_workflow,
    team_supervisor_workflow,
    team_supervisor_graph_workflow,
)


async def demo_intent_condition():
    """演示：意图识别 + 条件路由"""
    print("=" * 60)
    print("模式一：意图识别 + 条件路由")
    print("=" * 60)

    queries = [
        "帮我写一个 Python 快速排序",
        "最近有什么AI新闻",
        "你好，今天天气怎么样",
    ]

    for query in queries:
        print(f"\n[Query] {query}")
        result = await intent_condition_workflow(query)
        print(f"  意图: {result['intent']} (置信度: {result['confidence']:.2f})")
        print(f"  理由: {result['reason']}")
        print(f"  路由到: {result['routed_to']}")
        print(f"  输出: {result['output'][:200]}...")


async def demo_team_supervisor():
    """演示：Team Supervisor Python 原生版"""
    print("\n" + "=" * 60)
    print("模式二：Team Supervisor（Python 循环版）")
    print("=" * 60)

    query = "帮我查一下 Python 3.12 的新特性，然后写一个示例代码"
    print(f"\n[Query] {query}")
    result = await team_supervisor_workflow(query, max_rounds=2)

    print(f"\n  调用历史 ({len(result['history'])} 轮):")
    for step in result["history"]:
        print(f"    Round {step['round']}: {step['agent']}")
        print(f"    Output: {step['output'][:150]}...")

    print(f"\n  [Final Output]\n{result['final_output'][:300]}...")


async def demo_team_supervisor_graph():
    """演示：Team Supervisor LangGraph 版"""
    print("\n" + "=" * 60)
    print("模式三：Team Supervisor（LangGraph 版）")
    print("=" * 60)

    query = "帮我查一下 Python 3.12 的新特性，然后写一个示例代码"
    print(f"\n[Query] {query}")
    result = await team_supervisor_graph_workflow(query, max_rounds=2)

    print(f"\n  调用历史 ({len(result['history'])} 轮):")
    for step in result["history"]:
        print(f"    Round {step['round']}: {step['agent']}")
        print(f"    Output: {step['output'][:150]}...")

    print(f"\n  [Final Output]\n{result['final_output'][:300]}...")


async def demo_mixed_llm():
    """演示：不同 Agent 使用不同模型"""
    print("\n" + "=" * 60)
    print("模式四：混合模型（每个 Agent 独立 LLM）")
    print("=" * 60)

    from core.llm.model_type import ModelType
    from core.agents.specialized import CodeAgent, ChatAgent

    code_agent = CodeAgent(model_type=ModelType.QWEN_MAX)
    chat_agent = ChatAgent(model_type=ModelType.GPT_4O_MINI)
    coordinator_model = ModelType.GPT_4O

    query = "帮我写个 Python 快排，再闲聊两句"
    print(f"\n[Query] {query}")
    print(f"  CodeAgent:   {code_agent.model_type.value}")
    print(f"  ChatAgent:   {chat_agent.model_type.value}")
    print(f"  Coordinator: {coordinator_model.value}")

    result = await team_supervisor_graph_workflow(
        query,
        agents=[code_agent, chat_agent],
        supervisor_model=coordinator_model,
        max_rounds=2,
    )

    print(f"\n  调用历史 ({len(result['history'])} 轮):")
    for step in result["history"]:
        print(f"    Round {step['round']}: {step['agent']}")
    print(f"\n  [Final Output]\n{result['final_output'][:300]}...")


async def demo_factory_api():
    """演示：工厂 API 和动态注册"""
    print("\n" + "=" * 60)
    print("模式五：工厂 API 与动态注册")
    print("=" * 60)

    from core.llm.factory import llm_factory
    from core.llm.model_type import ModelType

    print(f"\n  已注册模型: {llm_factory.list_models()}")

    model = llm_factory.get_model(ModelType.GPT_4O_MINI)
    print(f"  获取模型 {ModelType.GPT_4O_MINI.value}: {model}")

    model2 = llm_factory.get_model("openai-gpt-4o")
    print(f"  获取模型 'openai-gpt-4o': {model2}")

    from langchain_openai import ChatOpenAI
    custom_model = ChatOpenAI(model="gpt-4o", temperature=0.5)
    llm_factory.register_model(ModelType.GPT_4O, custom_model)
    print(f"  动态注册完成: GPT_4O 已替换")


async def demo_custom_agent():
    """演示：如何扩展自定义 Agent"""
    print("\n" + "=" * 60)
    print("模式六：扩展自定义 Agent")
    print("=" * 60)

    from core.llm.model_type import ModelType
    from core.agents.base import Agent
    from core.routing.supervisor import TeamSupervisor

    class MyCustomAgent(Agent):
        name = "custom"
        system_prompt = "你是一个翻译助手，只负责把用户输入翻译成英文。"
        model_type = ModelType.GPT_4O_MINI

    agent = MyCustomAgent()
    output = await agent.run("你好世界")
    print(f"\n  CustomAgent 输出: {output}")

    supervisor = TeamSupervisor(agents=[agent], max_rounds=1)
    result = await supervisor.run("今天天气真好")
    print(f"  Supervisor 调用: {result['final_output']}")


async def main():
    await demo_intent_condition()
    await demo_team_supervisor()
    await demo_team_supervisor_graph()
    await demo_mixed_llm()
    await demo_factory_api()
    await demo_custom_agent()


if __name__ == "__main__":
    asyncio.run(main())
