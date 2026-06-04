"""
MCP Demo Server（金融数据版）
===========================
基于 hk-finance-mcp 的设计模式提炼的演示级 MCP Server。

工具列表（对齐 hk-finance-mcp 真实工具集）：
    - get_current_datetime      : 获取当前日期时间
    - get_valid_tables          : 获取可用数据表列表
    - get_table_info            : 获取表结构信息
    - query_by_stock_code       : 按股票代码查询
    - query_by_sponsor          : 按保荐人查询
    - filter_by_date            : 按日期范围筛选
    - filter_by_numeric_value   : 按数值条件筛选
    - find_official_company_name: 公司简称→官方全称模糊匹配
    - query_institution_main_business: 机构主营业务语义检索
    - calculate                 : 安全数学计算

用法:
    # stdio 模式（供本地 Agent 子进程调用）
    python -m mcp_bridge.server.demo_server --transport stdio

    # sse 模式（供远程 Agent HTTP 连接）
    python -m mcp_bridge.server.demo_server --transport sse --port 18080

    # 兼容 MCP Hub 格式测试
    python -m mcp_bridge.server.demo_server --transport sse --port 18080 --hub-compatible
"""
import argparse
import ast
import functools
import operator
import re
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional

# 先加载项目统一日志配置，接管 mcp/fastmcp 库的日志输出
from core.utils.logger import get_logger

logger = get_logger("mcp_bridge.server.demo_server")

from mcp.server.fastmcp import FastMCP
from pydantic import Field

# ---------------------------------------------------------------------------
# MCP Hub 兼容补丁（跟 hk-finance-mcp 完全一致的设计）
# ---------------------------------------------------------------------------
# FastMCP 默认只支持标准 MCP 格式：{"tool": "...", "arguments": {...}}
# MCP Hub / 某些网关会包装为：{"toolName": "...", "arguments": {...}}
# 通过 Monkey Patch 让 Server 同时兼容两种格式，无需上层修改。

_HUB_COMPATIBLE = False  # 由 CLI 参数控制是否启用

if _HUB_COMPATIBLE:
    import fastmcp.tools.tool_manager

    _original_call_tool = fastmcp.tools.tool_manager.ToolManager.call_tool

    async def _patched_call_tool(self, key: str, arguments: dict[str, Any]):
        """兼容 MCP Hub 的请求格式：解包嵌套参数并处理工具名前缀。"""
        if "toolName" in arguments and "arguments" in arguments:
            actual_tool_name = arguments["toolName"]
            actual_arguments = arguments["arguments"]

            # 处理工具名前缀（如 "hk-finance-db-query_by_stock_code" → "query_by_stock_code"）
            prefix = "hk-finance-db-"
            if actual_tool_name.startswith(prefix):
                actual_tool_name = actual_tool_name[len(prefix):]
                logger.info(
                    f"[HubPatch] 剥离前缀: {arguments['toolName']} -> {actual_tool_name}"
                )

            key = actual_tool_name
            arguments = actual_arguments

        return await _original_call_tool(self, key, arguments)

    fastmcp.tools.tool_manager.ToolManager.call_tool = _patched_call_tool
    logger.info("[HubPatch] MCP Hub 兼容补丁已启用")


# ---------------------------------------------------------------------------
# FastMCP 实例
# ---------------------------------------------------------------------------
mcp = FastMCP("hk-finance-db-demo")


# ---------------------------------------------------------------------------
# 可观测性：统一工具调用日志装饰器（跟 hk-finance-mcp 的 log_tool_call 对齐）
# ---------------------------------------------------------------------------
def log_tool_call(func):
    """
    统一记录工具调用的名称、参数、执行时长及异常。
    用法: 放在 @mcp.tool() 下方，@log_tool_call 上方。
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        start_time = time.time()
        logger.info(f"[ToolCall] {func_name} | args={args} kwargs={kwargs}")

        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start_time
            logger.info(
                f"[ToolCall] {func_name} 成功 |耗时={elapsed:.4f}s| "
                f"result_type={type(result).__name__}"
            )
            return result
        except Exception as exc:
            elapsed = time.time() - start_time
            logger.error(
                f"[ToolCall] {func_name} 失败 |耗时={elapsed:.4f}s| error={exc}",
                exc_info=True,
            )
            # 统一返回错误字典，方便 Agent 侧做容错处理
            return {"error": f"Tool '{func_name}' execution failed: {exc}"}

    return wrapper


# ---------------------------------------------------------------------------
# Mock 数据库（内存字典，对齐 hk-finance-mcp 的真实表结构）
# ---------------------------------------------------------------------------

# 可用数据表定义（对齐 get_valid_tables 返回值）
_TABLE_SCHEMA = {
    "hk_basic_info": {
        "table_name_cn": "港股基本信息",
        "description": "Hong Kong stock basic information",
        "columns": [
            {"name": "stock_code", "type": "VARCHAR(10)", "comment": "股票代码"},
            {"name": "company_name", "type": "VARCHAR(255)", "comment": "公司全称（繁体中文）"},
            {"name": "listing_date", "type": "DATE", "comment": "上市日期"},
            {"name": "industry", "type": "VARCHAR(100)", "comment": "行业分类"},
            {"name": "market_cap", "type": "DECIMAL(20,2)", "comment": "市值（亿港元）"},
        ],
        "primary_key": ["stock_code"],
    },
    "hk_ipo_info": {
        "table_name_cn": "港股IPO信息",
        "description": "Hong Kong IPO purchase information",
        "columns": [
            {"name": "stock_code", "type": "VARCHAR(10)", "comment": "股票代码"},
            {"name": "company_name", "type": "VARCHAR(255)", "comment": "公司名称"},
            {"name": "sponsor", "type": "VARCHAR(255)", "comment": "保荐人"},
            {"name": "incorporation_place", "type": "VARCHAR(100)", "comment": "注册地（繁体中文）"},
            {"name": "currency", "type": "VARCHAR(10)", "comment": "货币单位", "default": "HKD"},
            {"name": "ipo_date", "type": "DATE", "comment": "上市日期"},
        ],
        "primary_key": ["stock_code"],
    },
    "hk_repurchase": {
        "table_name_cn": "港股回购信息",
        "description": "Hong Kong stock repurchase information",
        "columns": [
            {"name": "stock_code", "type": "VARCHAR(10)", "comment": "股票代码"},
            {"name": "repurchase_date", "type": "DATE", "comment": "回购日期"},
            {"name": "repurchase_shares", "type": "INT", "comment": "回购股数"},
            {"name": "repurchase_amount", "type": "DECIMAL(20,2)", "comment": "回购金额（港元）"},
        ],
        "primary_key": ["stock_code", "repurchase_date"],
    },
    "hk_institution": {
        "table_name_cn": "机构信息",
        "description": "Institution main business and profile",
        "columns": [
            {"name": "institution_id", "type": "VARCHAR(50)", "comment": "机构ID"},
            {"name": "institution_name", "type": "VARCHAR(255)", "comment": "机构名称"},
            {"name": "main_business", "type": "TEXT", "comment": "主营业务描述"},
        ],
        "primary_key": ["institution_id"],
    },
}

# Mock 数据记录（内存表）
_MOCK_DATA = {
    "hk_basic_info": [
        {
            "stock_code": "01810",
            "company_name": "小米集團－Ｗ",
            "listing_date": "2018-07-09",
            "industry": "資訊科技器材",
            "market_cap": 3850.50,
        },
        {
            "stock_code": "00700",
            "company_name": "騰訊控股有限公司",
            "listing_date": "2004-06-16",
            "industry": "軟件服務",
            "market_cap": 34200.00,
        },
        {
            "stock_code": "03690",
            "company_name": "美團－Ｗ",
            "listing_date": "2018-09-20",
            "industry": "電子商務及互聯網服務",
            "market_cap": 7200.00,
        },
        {
            "stock_code": "09888",
            "company_name": "百度集團－ＳＷ",
            "listing_date": "2021-03-23",
            "industry": "軟件服務",
            "market_cap": 2800.00,
        },
    ],
    "hk_ipo_info": [
        {
            "stock_code": "01810",
            "company_name": "小米集團－Ｗ",
            "sponsor": "摩根士丹利亞洲有限公司",
            "incorporation_place": "開曼群島",
            "currency": "HKD",
            "ipo_date": "2018-07-09",
        },
        {
            "stock_code": "09888",
            "company_name": "百度集團－ＳＷ",
            "sponsor": "美銀證券",
            "incorporation_place": "開曼群島",
            "currency": "HKD",
            "ipo_date": "2021-03-23",
        },
        {
            "stock_code": "06690",
            "company_name": "海爾智家股份有限公司",
            "sponsor": "中國國際金融香港證券有限公司",
            "incorporation_place": "中國",
            "currency": "HKD",
            "ipo_date": "2020-12-22",
        },
    ],
    "hk_repurchase": [
        {
            "stock_code": "00700",
            "repurchase_date": "2024-06-01",
            "repurchase_shares": 500000,
            "repurchase_amount": 185000000.00,
        },
        {
            "stock_code": "00700",
            "repurchase_date": "2024-06-15",
            "repurchase_shares": 480000,
            "repurchase_amount": 178000000.00,
        },
        {
            "stock_code": "01810",
            "repurchase_date": "2024-05-20",
            "repurchase_shares": 1200000,
            "repurchase_amount": 156000000.00,
        },
    ],
    "hk_institution": [
        {
            "institution_id": "INST_001",
            "institution_name": "中國國際金融股份有限公司",
            "main_business": "投資銀行業務、證券經紀、資產管理、研究諮詢",
        },
        {
            "institution_id": "INST_002",
            "institution_name": "摩根士丹利亞洲有限公司",
            "main_business": "環球投資銀行、證券交易、財富管理、研究分析",
        },
        {
            "institution_id": "INST_003",
            "institution_name": "美銀證券",
            "main_business": "環球市場交易、投資銀行、企業融資、股票研究",
        },
    ],
}


# ---------------------------------------------------------------------------
# 校验工具（对齐 hk-finance-mcp 的校验逻辑）
# ---------------------------------------------------------------------------

def _validate_table_name(table_name: str) -> bool:
    """校验表名是否在可用列表中。"""
    return table_name in _TABLE_SCHEMA


def _validate_date_format(date_str: str) -> bool:
    """校验日期格式是否为 YYYY-MM-DD。"""
    if not date_str:
        return False
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return True
    except ValueError:
        return False


def _is_traditional_hanzi(text: str) -> bool:
    """
    简单繁体中文检测（演示版）。
    生产环境中会用更完整的字符集校验。
    """
    if not text:
        return False
    # 常见简体字检测：如果包含明显简体字，返回 False
    simplified_chars = set("团级东专业丛丝丢两严临")
    for ch in text:
        if ch in simplified_chars:
            return False
    # 至少包含一个中文字符
    has_chinese = any("\u4e00" <= ch <= "\u9fff" for ch in text)
    return has_chinese


def _calculate_text_similarity(s1: str, s2: str) -> float:
    """
    编辑距离相似度（0.0-1.0，0.0 表示完全匹配，1.0 表示完全不匹配）。
    跟 hk-finance-mcp 的 calculate_text_similarity 对齐。
    """
    if not s1 or not s2:
        return 1.0

    s1, s2 = s1.strip().lower(), s2.strip().lower()
    if s1 == s2:
        return 0.0

    # Levenshtein 距离
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    max_len = max(m, n)
    return dp[m][n] / max_len if max_len > 0 else 0.0


# ---------------------------------------------------------------------------
# 工具实现（对齐 hk-finance-mcp 真实工具集，用 Mock 数据）
# ---------------------------------------------------------------------------

@mcp.tool(description="Get the current date and time")
@log_tool_call
def get_current_datetime(timezone: str = "Asia/Hong_Kong") -> str:
    """
    Get the current date and time in the specified timezone.

    Args:
        timezone: Timezone name (default: Asia/Hong_Kong)
    """
    try:
        import pytz
        tz = pytz.timezone(timezone)
        return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")


@mcp.tool(description="Get current valid table names with their descriptions")
@log_tool_call
def get_valid_tables() -> Dict[str, Dict[str, str]]:
    """
    Get the current list of valid table names along with their Chinese descriptions.

    Returns:
        Dict where keys are table names and values contain:
            - table_name_cn: Chinese name
            - description: Table description
    """
    return {
        name: {
            "table_name_cn": info["table_name_cn"],
            "description": info["description"],
        }
        for name, info in _TABLE_SCHEMA.items()
    }


@mcp.tool(description="Reflect and return table information, including columns, primary key, and comments")
@log_tool_call
def get_table_info(
    table_name: str = Field(description="The name of the table to get information for"),
) -> Dict:
    """
    Reflect table schema including columns, types, comments and primary keys.
    """
    if not _validate_table_name(table_name):
        return {
            "error": f"Invalid table name: {table_name}. "
            f"Available tables: {list(_TABLE_SCHEMA.keys())}"
        }

    info = _TABLE_SCHEMA[table_name]
    return {
        "table_name": table_name,
        "table_name_cn": info["table_name_cn"],
        "description": info["description"],
        "columns": info["columns"],
        "primary_key": info["primary_key"],
    }


@mcp.tool(
    description="Query by stock code (convenience function). Basically supports queries for all tables."
)
@log_tool_call
def query_by_stock_code(
    stock_code: str = Field(description="The stock code to query, e.g. 00700, 01810"),
    table_name: str = Field(description="The name of the table to query"),
    limit: int = Field(default=50, description="Maximum number of records to return"),
    offset: int = Field(default=0, description="Number of records to skip for pagination"),
) -> Dict:
    """
    Query records by stock code from the specified table.
    Supports pagination via offset/limit.
    """
    if not _validate_table_name(table_name):
        return {
            "error": f"Invalid table name: {table_name}. "
            f"Available: {list(_TABLE_SCHEMA.keys())}"
        }

    table_data = _MOCK_DATA.get(table_name, [])
    all_matches = [
        row for row in table_data
        if row.get("stock_code") == stock_code
    ]
    total_count = len(all_matches)
    results = all_matches[offset : offset + limit]

    return {
        "table_name": table_name,
        "stock_code": stock_code,
        "count": len(results),
        "total_count": total_count,
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(results) < total_count,
        "data": results,
    }


@mcp.tool(description="Query data by sponsor (保荐人)")
@log_tool_call
def query_by_sponsor(
    sponsor_name: str = Field(description="The sponsor name to query, e.g. 中金, 摩根士丹利"),
    table_name: str = Field(
        default="hk_ipo_info",
        description="The table to query (default: hk_ipo_info)",
    ),
    limit: int = Field(default=50, description="Maximum records to return"),
    offset: int = Field(default=0, description="Number of records to skip for pagination"),
) -> Dict:
    """
    Query IPO records by sponsor name.
    Supports pagination via offset/limit.
    """
    if not _validate_table_name(table_name):
        return {
            "error": f"Invalid table name: {table_name}. "
            f"Available: {list(_TABLE_SCHEMA.keys())}"
        }

    table_data = _MOCK_DATA.get(table_name, [])
    # 支持部分匹配（演示简化版，生产环境用更复杂的模糊匹配）
    all_matches = [
        row for row in table_data
        if sponsor_name in row.get("sponsor", "")
    ]
    total_count = len(all_matches)
    results = all_matches[offset : offset + limit]

    return {
        "table_name": table_name,
        "sponsor": sponsor_name,
        "count": len(results),
        "total_count": total_count,
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(results) < total_count,
        "data": results,
    }


@mcp.tool(description="Filter data by date range")
@log_tool_call
def filter_by_date(
    table_name: str = Field(description="The table to query"),
    date_column: str = Field(description="The date column name to filter on"),
    end_date: str = Field(description="End date in YYYY-MM-DD format"),
    start_date: Optional[str] = Field(default=None, description="Start date in YYYY-MM-DD format"),
    limit: int = Field(default=50, description="Maximum records to return"),
    offset: int = Field(default=0, description="Number of records to skip for pagination"),
) -> Dict:
    """
    Filter table records by a date range on the specified date column.
    Supports pagination via offset/limit.
    """
    if not _validate_table_name(table_name):
        return {
            "error": f"Invalid table name: {table_name}. "
            f"Available: {list(_TABLE_SCHEMA.keys())}"
        }

    if not _validate_date_format(end_date):
        return {"error": f"Invalid end_date format: {end_date}. Expected YYYY-MM-DD."}

    if start_date and not _validate_date_format(start_date):
        return {"error": f"Invalid start_date format: {start_date}. Expected YYYY-MM-DD."}

    table_data = _MOCK_DATA.get(table_name, [])
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None

    all_matches = []
    for row in table_data:
        row_date_str = row.get(date_column)
        if not row_date_str:
            continue
        try:
            row_dt = datetime.strptime(str(row_date_str), "%Y-%m-%d")
        except ValueError:
            continue

        if row_dt > end_dt:
            continue
        if start_dt and row_dt < start_dt:
            continue
        all_matches.append(row)

    total_count = len(all_matches)
    results = all_matches[offset : offset + limit]

    return {
        "table_name": table_name,
        "date_column": date_column,
        "start_date": start_date,
        "end_date": end_date,
        "count": len(results),
        "total_count": total_count,
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(results) < total_count,
        "data": results,
    }


@mcp.tool(description="Filter data by numeric column value with comparison operators")
@log_tool_call
def filter_by_numeric_value(
    table_name: str = Field(description="The table to query"),
    column_name: str = Field(description="The numeric column name to filter on"),
    operator: str = Field(description="Comparison operator: =, >, <, >=, <=, !="),
    value: Optional[float] = Field(default=None, description="The numeric value to compare"),
    limit: int = Field(default=50, description="Maximum records to return"),
    offset: int = Field(default=0, description="Number of records to skip for pagination"),
) -> Dict:
    """
    Filter table records by a numeric condition.
    Supports pagination via offset/limit.
    """
    if not _validate_table_name(table_name):
        return {
            "error": f"Invalid table name: {table_name}. "
            f"Available: {list(_TABLE_SCHEMA.keys())}"
        }

    valid_operators = {"=", ">", "<", ">=", "<=", "!="}
    if operator not in valid_operators:
        return {
            "error": f"Invalid operator: {operator}. Must be one of {valid_operators}"
        }

    if value is None:
        return {"error": "Value cannot be None for numeric comparison."}

    table_data = _MOCK_DATA.get(table_name, [])

    ops = {
        "=": lambda a, b: a == b,
        ">": lambda a, b: a > b,
        "<": lambda a, b: a < b,
        ">=": lambda a, b: a >= b,
        "<=": lambda a, b: a <= b,
        "!=": lambda a, b: a != b,
    }
    op_func = ops[operator]

    all_matches = []
    for row in table_data:
        row_val = row.get(column_name)
        if row_val is None:
            continue
        try:
            row_num = float(row_val)
        except (ValueError, TypeError):
            continue

        if op_func(row_num, value):
            all_matches.append(row)

    total_count = len(all_matches)
    results = all_matches[offset : offset + limit]

    return {
        "table_name": table_name,
        "column_name": column_name,
        "operator": operator,
        "value": value,
        "count": len(results),
        "total_count": total_count,
        "offset": offset,
        "limit": limit,
        "has_more": offset + len(results) < total_count,
        "data": results,
    }


@mcp.tool(
    description="Find official company names by searching for similar names in hk_basic_info table. "
                "Input company abbreviations or partial names to get official registered names. "
                "Must use Traditional Chinese characters."
)
@log_tool_call
def find_official_company_name(
    query: str = Field(
        description="Company abbreviation or partial name (Traditional Chinese), e.g. 小米, 騰訊"
    ),
    limit: int = Field(default=3, description="Maximum results (1-5)"),
    similarity_threshold: float = Field(
        default=0.4, description="Similarity threshold (0.0-1.0, lower is more similar)"
    ),
) -> Dict:
    """
    通过公司简称或部分名称查找官方注册的公司全称。
    使用编辑距离算法计算相似度，支持繁体中文。
    """
    if not query or not query.strip():
        return {"error": "Query cannot be empty"}

    if not (0.0 <= similarity_threshold <= 1.0):
        return {"error": "Similarity threshold must be between 0.0 and 1.0"}

    if not (1 <= limit <= 5):
        return {"error": "Limit must be between 1 and 5"}

    if not _is_traditional_hanzi(query):
        return {
            "error": "Company name query should use Traditional Chinese characters "
            "(e.g. 騰訊 instead of 腾讯)."
        }

    candidates = []
    for row in _MOCK_DATA.get("hk_basic_info", []):
        company_name = row.get("company_name", "")
        sim = _calculate_text_similarity(query, company_name)
        if sim <= similarity_threshold:
            candidates.append({
                "stock_code": row.get("stock_code"),
                "company_name": company_name,
                "similarity_score": round(sim, 4),
            })

    candidates.sort(key=lambda x: x["similarity_score"])
    results = candidates[:limit]

    return {
        "query": query,
        "count": len(results),
        "results": results,
    }


@mcp.tool(description="Query institution main business via semantic search")
@log_tool_call
def query_institution_main_business(
    query: str = Field(description="The query to search for institution main business, e.g. 投資銀行"),
    k: int = Field(default=5, description="Top K results to return"),
) -> List[Dict]:
    """
    语义检索机构主营业务（演示版）。
    生产环境会调用 Qdrant 向量库 + Voyage Rerank 做 Embedding 相似度搜索。
    本 Demo 使用关键词匹配模拟语义检索效果。
    """
    if not query or not query.strip():
        return [{"error": "Query cannot be empty"}]

    query_lower = query.lower()
    table_data = _MOCK_DATA.get("hk_institution", [])

    # 模拟语义检索：计算 query 与主营业务描述的匹配度
    scored = []
    for row in table_data:
        business = row.get("main_business", "")
        # 简单关键词匹配 + 编辑距离综合评分
        keywords = set(query_lower.split())
        business_lower = business.lower()
        keyword_hits = sum(1 for kw in keywords if kw in business_lower)
        edit_sim = _calculate_text_similarity(query, business)
        # 综合分数：关键词命中越高越好，编辑距离越低越好
        score = keyword_hits * 0.3 + (1 - edit_sim) * 0.7
        scored.append({"row": row, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    top_results = [
        {
            "institution_id": item["row"]["institution_id"],
            "institution_name": item["row"]["institution_name"],
            "main_business": item["row"]["main_business"],
            "similarity_score": round(item["score"], 4),
        }
        for item in scored[:k]
    ]

    return top_results


# ---------------------------------------------------------------------------
# 保留原设计：安全数学计算工具
# ---------------------------------------------------------------------------
_SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}


def _safe_eval(node: ast.AST) -> Any:
    """递归解析 AST，仅允许常量与基本四则运算。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        return _SAFE_OPS[op_type](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        return _SAFE_OPS[op_type](_safe_eval(node.operand))
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    raise ValueError(f"Unsupported expression node: {type(node).__name__}")


@mcp.tool(description="Safely evaluate a mathematical expression")
@log_tool_call
def calculate(expression: str) -> str:
    """
    安全计算数学表达式（AST 白名单解析，无 eval 风险）。

    Supported: +, -, *, /, //, %, **, parentheses.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree)
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero."
    except Exception as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="MCP Demo Server (Finance Data Edition)")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="Transport type: stdio (default) or sse (HTTP streaming)",
    )
    parser.add_argument("--host", default="127.0.0.1", help="SSE listen address")
    parser.add_argument("--port", type=int, default=18080, help="SSE listen port")
    parser.add_argument(
        "--hub-compatible",
        action="store_true",
        help="Enable MCP Hub compatible format (requires restart to take effect)",
    )
    args = parser.parse_args()

    if args.hub_compatible:
        logger.info(
            "[CLI] --hub-compatible 已指定，但补丁需要在导入时生效。"
            "请在代码中将 _HUB_COMPATIBLE 设为 True 后重新启动。"
        )

    if args.transport == "sse":
        logger.info(
            f"[DemoServer] 启动 SSE 模式 | http://{args.host}:{args.port}/sse"
        )
    else:
        logger.info("[DemoServer] 启动 stdio 模式")

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
