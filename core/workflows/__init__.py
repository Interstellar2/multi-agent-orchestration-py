"""
工作流组合模块

将意图识别、路由、Agent 调用组合成完整的工作流。

核心设计：通过 ModelType 枚举 + 工厂统一管理 LLM，
每个组件（Classifier / Supervisor / Agent）可独立指定模型类型。

用法:
    from core.workflows import intent_condition_workflow, team_supervisor_graph_workflow

    # 每个 Agent 配不同模型
    code_agent = CodeAgent(model_type=ModelType.QWEN_MAX)
    chat_agent = ChatAgent(model_type=ModelType.GPT_4O_MINI)

    # Supervisor 用最强的模型做决策
    result = await team_supervisor_graph_workflow(
        query,
        agents=[code_agent, chat_agent],
        supervisor_model=ModelType.GPT_4O,
    )
"""
from core.workflows.intent_condition import intent_condition_workflow
from core.workflows.supervisor import team_supervisor_workflow, team_supervisor_graph_workflow
from core.workflows.mcp import mcp_react_workflow, mcp_supervisor_workflow

# 资本市场工作流（委托给业务域，保持向后兼容）
async def capital_market_research_workflow(*args, **kwargs):
    """模式六：资本市场研究助理（向后兼容导出）"""
    from domains.capital_market.main import capital_market_research_workflow as _wf
    return await _wf(*args, **kwargs)


async def capital_market_team_workflow(*args, **kwargs):
    """模式七：Team Supervisor + 资本市场研究助理（向后兼容导出）"""
    from domains.capital_market.main import capital_market_team_workflow as _wf
    return await _wf(*args, **kwargs)


# 保留公共 API（迁移前被外部使用的私有函数已移到 core.agents.resolver）
from core.agents.resolver import resolve_agents

__all__ = [
    "intent_condition_workflow",
    "team_supervisor_workflow",
    "team_supervisor_graph_workflow",
    "mcp_react_workflow",
    "mcp_supervisor_workflow",
    "capital_market_research_workflow",
    "capital_market_team_workflow",
    "resolve_agents",
]
