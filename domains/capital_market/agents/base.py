"""
资本市场研究助理 Agent
通过 MCP 协议连接 hk-finance-mcp Server，查询港交所金融数据。

使用方式:
    # 方式一：默认连接本地 hk-finance-mcp SSE 服务
    agent = CapitalMarketAgent()

    # 方式二：指定其他 MCP Server
    agent = CapitalMarketAgent(server_url="http://127.0.0.1:18080/sse")

    # 方式三：使用 stdio 启动本地 Server
    agent = CapitalMarketAgent(
        server_cmd=["python", "/path/to/hk-finance-mcp/main.py"]
    )

    result = await agent.run("查询腾讯控股最近一年的回购记录")
"""
from typing import List, Optional

from langchain_core.language_models.chat_models import BaseChatModel

from core.agents.base import Agent
from core.agents.toolkit import MCPClientProvider
from core.llm.model_type import ModelType
from core.utils.logger import get_logger

logger = get_logger(__name__)


class CapitalMarketAgent(Agent):
    """
    香港资本市场研究助理。

    继承 Agent 统一基座，组合 MCPClientProvider 实现 MCP 能力。
    默认连接 hk-finance-mcp 来获取港交所金融数据（text2sql、向量检索、
    公司名称模糊匹配等）。

    与通用 MCPAgent 的区别：
      1. 带有金融业务身份和领域知识（system_prompt）
      2. 默认指向真实金融 MCP Server，而非 demo_server
      3. 可被 Supervisor 作为团队中"金融数据专家"调度
    """

    name = "capital_market"
    system_prompt = (
        "你是一名香港资本市场研究助理，精通港交所上市公司、IPO、回购、"
        "机构业务等金融数据的查询与分析。\n\n"
        "你的工作流程：\n"
        "1. 理解用户的金融数据需求（如查股票、查公司、查保荐人、查回购等）\n"
        "2. 通过 MCP 工具查询 hk-finance-mcp 数据库获取准确数据\n"
        "3. 对查询结果进行解读、对比和简要分析\n"
        "4. 使用繁体中文处理香港公司名称查询\n\n"
        "查询策略建议：\n"
        "- 首次查询前，可用 get_valid_tables 了解可用数据表，"
        "再用 get_table_info 查看表结构\n"
        "- 按股票代码查询优先使用 query_by_stock_code\n"
        "- 用户输入公司简称时，先用 find_official_company_name 获取官方全称\n"
        "- 日期范围筛选使用 filter_by_date，数值条件使用 filter_by_numeric_value\n"
        "- 机构主营业务语义检索使用 query_institution_main_business\n"
        "- 如查询失败，向用户说明原因并建议调整查询条件\n\n"
        "注意事项：\n"
        "- 你只能查询数据库中已有的数据，不做预测或投资建议\n"
        "- 涉及公司名称时，注意香港公司使用繁体中文"
    )
    model_type = ModelType.GPT_4O

    # hk-finance-mcp 默认 SSE 地址
    DEFAULT_SERVER_URL = "http://127.0.0.1:1888/mcp/sse"

    def __init__(
        self,
        model_type: Optional[ModelType] = None,
        llm: Optional[BaseChatModel] = None,
        server_url: Optional[str] = None,
        server_cmd: Optional[List[str]] = None,
    ):
        """
        初始化资本市场研究助理。

        参数:
            model_type: LLM 模型类型（覆盖默认值）
            llm: 直接传入 LLM 实例（最高优先级）
            server_url: MCP Server SSE 地址，默认 http://127.0.0.1:1888/mcp/sse
            server_cmd: stdio 启动命令（与 server_url 二选一）

        注意:
            使用默认地址前，请确保 hk-finance-mcp 服务已启动：
                cd /Users/hanniandong/python_project/hk-finance-mcp
                python main.py   # 或 uvicorn main:app --port 1888
        """
        if not server_url and not server_cmd:
            server_url = self.DEFAULT_SERVER_URL
            logger.info(
                f"[{self.name}] 未指定 MCP Server，"
                f"使用默认 hk-finance-mcp SSE: {server_url}"
            )

        provider = MCPClientProvider(server_url=server_url, server_cmd=server_cmd)
        super().__init__(model_type=model_type, llm=llm, tools=[provider])
