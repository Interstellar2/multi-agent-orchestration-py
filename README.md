# Multi-Agent + MCP Demo

一个轻量级的多 Agent 协作与 MCP 协议集成示例项目，纯 Python 实现。

支持 **七种协作模式**：
- **意图识别 + 条件路由**：LLM 识别意图，规则路由到固定 Agent
- **Team Supervisor（循环版）**：LLM 动态决策调用哪个 Agent，串行执行，上下文传递
- **Team Supervisor（LangGraph 版）**：同上，基于 LangGraph `Command(goto=...)` 实现
- **MCP ReAct Agent**：单 Agent 通过 MCP 协议发现并调用外部工具
- **MCP Supervisor**：Supervisor 调度 MCP-enabled Agent，实现多 Agent + 外部工具的协作
- **资本市场研究助理（独立）**：带金融业务身份的 MCP Agent，直连港交所金融数据
- **资本市场团队（Supervisor）**：Supervisor 协调 ChatAgent + CapitalMarketAgent，自动路由金融/非金融问题

核心设计：
- **通用框架** (`core/`)：Agent 基类、LLM 工厂、多种路由/协调模式
- **业务域** (`domains/`)：垂直业务应用
  - `hk_law/`：香港法律多 Agent 系统，每个法域独立 Agent + ES RAG
  - `capital_market/`：香港资本市场研究助理（MCP + 金融业务身份）
- **MCP 集成** (`mcp_bridge/`)：FastMCP Server + MCPAgent，演示 LLM 与外部金融工具的标准协议通信

---

## 安装

```bash
pip install -r requirements.txt
```

---

## 配置 API Key

支持三种方式，优先级从高到低：

### 方式一：.env 文件（推荐）

在项目根目录创建 `.env`：

```bash
# LLM API Key（至少配一个）
OPENAI_API_KEY=sk-xxx
DASHSCOPE_API_KEY=sk-yyy          # 百炼 / 阿里云（Embedding + Rerank 默认使用）
DEEPSEEK_API_KEY=sk-zzz
KIMI_API_KEY=sk-aaa               # Moonshot
OPENROUTER_API_KEY=sk-bbb         # OpenRouter 统一网关

# Elasticsearch（默认 localhost:9200，如无密码可不配）
ELASTICSEARCH_URL=http://localhost:9200

# --- 本地部署（可选）---
# 使用 Ollama 本地 Embedding 时设置以下变量，无需 DASHSCOPE_API_KEY
# EMBEDDING_PROVIDER=ollama
# RERANK_PROVIDER=local
# OLLAMA_HOST=http://localhost:11434
# OLLAMA_EMBED_MODEL=nomic-embed-text
# OLLAMA_EMBED_DIMS=768
```

程序启动时会**自动加载** `.env` 中的环境变量。

### 方式二：config.yaml

编辑项目自带的 `config.yaml`：

```yaml
providers:
  openai:
    api_key: ${OPENAI_API_KEY}
    base_url: https://api.openai.com/v1
  bailian:
    api_key: ${DASHSCOPE_API_KEY}
    base_url: https://dashscope.aliyuncs.com/compatible-mode/v1
  deepseek:
    api_key: ${DEEPSEEK_API_KEY}
    base_url: https://api.deepseek.com/v1
  kimi:
    api_key: ${KIMI_API_KEY}
    base_url: https://api.moonshot.cn/v1
  openrouter:
    api_key: ${OPENROUTER_API_KEY}
    base_url: https://openrouter.ai/api/v1
```

`${ENV_VAR}` 语法会自动替换为对应环境变量值。

### 方式三：直接设置环境变量

```bash
export OPENAI_API_KEY=sk-xxx
export DASHSCOPE_API_KEY=sk-yyy
python main.py
```

---

## 快速开始

### 1. 资本市场研究助理（连接 hk-finance-mcp）

```bash
# 独立模式：CapitalMarketAgent 直连港交所金融数据
python main.py capital_market --mode research

# 团队模式：Supervisor 自动路由金融/非金融问题
python main.py capital_market --mode team

# 连接自定义 MCP Server（如真实 hk-finance-mcp）
python main.py capital_market --mode research --server-url http://127.0.0.1:1888/mcp/sse
```

### 2. MCP 工具调用演示

```bash
# MCP ReAct 模式：单 Agent 调用外部工具（Mock 金融数据）
python main.py mcp --mode react --transport stdio

# MCP + Supervisor 模式：Supervisor 动态调度 MCP Agent
python main.py mcp --mode supervisor --transport stdio

# SSE 模式（需先手动启动 MCP Server）
python -m mcp_bridge.server.demo_server --transport sse --port 18080
python main.py mcp --mode react --transport sse --server-url http://127.0.0.1:18080/sse
```

### 2. 通用多 Agent Demo

```bash
python -m core.demo
```

输出包含 6 个演示场景：
1. 意图识别 + 条件路由
2. Team Supervisor（循环版，串行上下文传递）
3. Team Supervisor（LangGraph 版，串行上下文传递）
4. 混合模型（不同 Agent 用不同 LLM）
5. 工厂 API 与动态注册
6. 扩展自定义 Agent

### 3. 香港法律多 Agent 系统

```bash
# 交互式问答
python main.py hk_law --mode interactive

# 或运行 Demo 测试用例
python main.py hk_law --mode demo

# 也可以直接调用模块
python -m domains.hk_law.main --mode interactive
```

---

## 香港法律多 Agent 系统

本项目包含一个完整的 **香港法律多 Agent 系统**，每个法域对应独立的 Agent，基于 Elasticsearch 提供 RAG 能力。Embedding 和 Rerank 默认使用百炼云端模型，也支持切换到 Ollama 本地模型（零 API Key 成本）。

### 1. 部署基础设施

```bash
docker-compose up -d
```

启动后：
- Elasticsearch: http://localhost:9200
- Kibana: http://localhost:5601
- MCP Demo Server: http://localhost:18080/sse

### 2. 下载法律文档

从香港律政司 [电子版香港法例](https://www.elegislation.gov.hk/) 下载以下条例的 PDF/Word，放入对应目录：

| 法域 | 目录 | 建议下载条例 |
|------|------|-------------|
| 刑事 | `domains/hk_law/documents/criminal/` | 《刑事罪行条例》(Cap. 200)、《盗窃罪条例》(Cap. 210) |
| 民事 | `domains/hk_law/documents/civil/` | 《合约(第三方权利)条例》(Cap. 623)、《失实陈述条例》(Cap. 284) |
| 公司 | `domains/hk_law/documents/company/` | 《公司条例》(Cap. 622) |
| 雇佣 | `domains/hk_law/documents/employment/` | 《雇佣条例》(Cap. 57) |
| 物业 | `domains/hk_law/documents/property/` | 《物业转易及财产条例》(Cap. 219)、《建筑物管理条例》(Cap. 344) |

下载地址：https://www.elegislation.gov.hk/

### 3. 构建向量索引

```bash
# 索引单个法域
python -m domains.hk_law.rag.ingest criminal

# 索引所有法域
python -m domains.hk_law.rag.ingest --all

# 重建索引（先删除再重建）
python -m domains.hk_law.rag.ingest criminal --rebuild
```

索引流程：
1. 加载 `documents/<domain>/` 下的文档
2. 切分为片段
3. 用 Embedding 模型生成向量（默认百炼 `text-embedding-v3`，或 Ollama 本地模型）
4. 写入 Elasticsearch `hk_law_<domain>` index

#### 本地部署（无需百炼 API Key）

如果你有 Ollama，可以用本地模型替代百炼：

```bash
# 1. 安装并启动 Ollama（https://ollama.com）
# 2. 拉取 embedding 模型
ollama pull nomic-embed-text

# 3. 在 .env 中切换 provider
export EMBEDDING_PROVIDER=ollama
export RERANK_PROVIDER=local
export OLLAMA_HOST=http://localhost:11434

# 4. 构建索引（与上面命令相同）
python -m domains.hk_law.rag.ingest --all
```

### 4. 运行法律问答

```bash
# 交互式问答
python main.py hk_law --mode interactive

# 或运行 Demo 测试用例
python main.py hk_law --mode demo

# 也可以直接调用模块
python -m domains.hk_law.main --mode interactive
```

或使用 API：

```python
from domains.hk_law.main import HKLawSystem

system = HKLawSystem()

# 意图识别 + 条件路由（适合单法域问题）
result = await system.ask("我被公司无故解雇了", mode="intent")

# Supervisor 动态协调（适合交叉法域问题，串行上下文传递）
result = await system.ask("我被公司解雇还欠工资", mode="supervisor")

print(result["output"])
```

---

## MCP 协议集成（新增）

本项目演示了 **Model Context Protocol (MCP)** 的完整集成链路：

### 架构

```
User Query
    |
    v
[MCPAgent / Supervisor]
    |
    v
[MCP Client] --(stdio/SSE)--> [MCP Server (FastMCP)]
    |                                    |
    v                                    v
[LLM ReAct Loop]                   [Tools]
    |                                - get_valid_tables
    v                                - get_table_info
[Final Answer]                     - query_by_stock_code
                                   - query_by_sponsor
                                   - filter_by_date
                                   - filter_by_numeric_value
                                   - find_official_company_name
                                   - query_institution_main_business
                                   - calculate
```

### 使用方式

```python
from core.agents.mcp_agent import MCPAgent
from core.llm.model_type import ModelType

# stdio 模式：自动启动本地 MCP Server
agent = MCPAgent(
    server_cmd=["python", "-m", "mcp_bridge.server.demo_server", "--transport", "stdio"],
    model_type=ModelType.GPT_4O_MINI,
)
result = await agent.run("查询小米集团的股票代码")

# SSE 模式：连接远程 MCP Server
agent = MCPAgent(
    server_url="http://127.0.0.1:18080/sse",
    model_type=ModelType.GPT_4O_MINI,
)
result = await agent.run("帮我计算 (3+5)*2")
```

### MCP 工作流

```python
from core.workflows import mcp_react_workflow, mcp_supervisor_workflow

# 单 Agent MCP ReAct（Mock 金融数据）
result = await mcp_react_workflow("查询小米集团的股票代码和上市信息")

# Supervisor + MCP Agent（多 Agent + 外部工具协作）
result = await mcp_supervisor_workflow(
    "查一下腾讯的回购记录，顺便算一下 100 除以 4",
    server_cmd=["python", "-m", "mcp_bridge.server.demo_server"],
)
```

### MCP Server 工具列表

#### 金融数据版（`demo_server.py`，对齐 `hk-finance-mcp` 设计模式）

| 工具 | 说明 | 输入 |
|------|------|------|
| `get_valid_tables` | 获取可用数据表列表 | 无 |
| `get_table_info` | 获取表结构（列名、类型、主键、注释） | `table_name: str` |
| `query_by_stock_code` | 按股票代码查询（支持多表） | `stock_code, table_name, limit` |
| `query_by_sponsor` | 按保荐人查询 IPO 信息 | `sponsor_name, table_name, limit` |
| `filter_by_date` | 按日期范围筛选 | `table_name, date_column, start_date, end_date` |
| `filter_by_numeric_value` | 按数值条件筛选（`>`, `<`, `=`, `!=` 等） | `table_name, column_name, operator, value` |
| `find_official_company_name` | 公司简称→官方全称（编辑距离 + 繁体中文校验） | `query, limit, similarity_threshold` |
| `query_institution_main_business` | 机构主营业务语义检索（Mock 向量检索） | `query, k` |
| `calculate` | 安全数学计算（AST 白名单，无 eval） | `expression: str` |

> 所有工具数据均为 Mock，无外部 API 调用，零业务敏感信息。工具参数均使用 Pydantic `Field(description=...)` 描述，LLM 可精准理解工具语义。

#### 连接真实 `hk-finance-mcp`

```bash
# 1. 启动真实金融 MCP Server（需配置 database_url）
cd /path/to/hk-finance-mcp
python main.py   # SSE 模式，默认 1888 端口

# 2. 连接真实服务
python main.py capital_market --mode research --server-url http://127.0.0.1:1888/mcp/sse
```

---

## 资本市场研究助理

**资本市场研究助理**（`CapitalMarketAgent`）是本项目 MCP 协议集成的核心业务 Agent，专门负责查询港交所金融数据。它继承自统一 Agent 基座并组合 `MCPClientProvider`，默认连接 `hk-finance-mcp`（或本项目 Mock 版 MCP Server）。

### 设计定位

与通用 `MCPAgent` 的区别：
1. **金融业务身份**：带有港交所金融数据专家的 system_prompt，指导 LLM 如何拆解用户意图、选择正确工具
2. **默认指向真实服务**：`DEFAULT_SERVER_URL = http://127.0.0.1:1888/mcp/sse`，直接对接 `hk-finance-mcp`
3. **可被 Supervisor 调度**：注册到 Agent 工厂后，Supervisor 团队中的"金融专家"角色

### 使用方式

```python
from domains.capital_market.agents import CapitalMarketAgent
from core.llm.model_type import ModelType

# 默认连接 hk-finance-mcp（需先启动真实服务）
agent = CapitalMarketAgent()
result = await agent.run("查询小米集团最近一年的回购记录")

# 连接 Mock MCP Server（零外部依赖）
agent = CapitalMarketAgent(server_url="http://127.0.0.1:18080/sse")
result = await agent.run("查一下腾讯的股票代码")

# 自定义模型
agent = CapitalMarketAgent(model_type=ModelType.QWEN_MAX)
```

### 工作流封装

```python
from core.workflows import (
    capital_market_research_workflow,  # 独立运行
    capital_market_team_workflow,      # Supervisor 团队协作
)

# 模式六：单独跑
result = await capital_market_research_workflow(
    "查询中金保荐了哪些公司",
    server_url="http://127.0.0.1:1888/mcp/sse",
)

# 模式七：Supervisor 团队中作为金融专家
result = await capital_market_team_workflow(
    "小米的回购数据是多少？顺便聊聊它的行业地位",
    server_url="http://127.0.0.1:1888/mcp/sse",
)
# Supervisor 会自动判断：金融问题 -> CapitalMarketAgent，闲聊 -> ChatAgent
```

### MCP Server 设计演进

本项目的 `mcp_bridge/server/demo_server.py` 是基于 `hk-finance-mcp` 生产级设计模式提炼的演示版：

| 设计要素 | `hk-finance-mcp`（生产级） | `demo_server.py`（演示版） |
|----------|---------------------------|---------------------------|
| FastMCP + `@mcp.tool()` | ✅ | ✅ |
| Pydantic `Field(description=...)` | ✅ | ✅ |
| 统一 `@log_tool_call` 装饰器 | ✅ | ✅ |
| 严格输入校验（表名、日期、数值、繁体中文） | ✅ | ✅ |
| MCP Hub 兼容补丁 | ✅ | ✅（可选） |
| 编辑距离文本匹配 | ✅ | ✅ |
| 真实数据库（SQLAlchemy + Qdrant） | ✅ | ❌（Mock 内存数据） |
| Voyage AI Rerank | ✅ | ❌（模拟语义检索） |

> Mock 数据包含 4 张表（`hk_basic_info`、`hk_ipo_info`、`hk_repurchase`、`hk_institution`），覆盖股票代码、IPO 保荐、回购记录、机构业务等典型场景。零外部依赖，面试演示直接跑。

---

## 核心用法

### 1. 指定模型类型

通过 `ModelType` 枚举选择 LLM，API Key 自动从配置读取：

```python
from core.llm.model_type import ModelType
from core.agents.specialized import CodeAgent

agent = CodeAgent(model_type=ModelType.QWEN_MAX)
output = await agent.run("写个 Python 快排")
```

### 2. 混合模型（每个 Agent 不同 LLM）

```python
from core.llm.model_type import ModelType
from core.agents.specialized import CodeAgent, ChatAgent
from core.workflows import team_supervisor_graph_workflow

code_agent = CodeAgent(model_type=ModelType.QWEN_MAX)
chat_agent = ChatAgent(model_type=ModelType.GPT_4O_MINI)

result = await team_supervisor_graph_workflow(
    "帮我写个快排，再闲聊两句",
    agents=[code_agent, chat_agent],
    supervisor_model=ModelType.GPT_4O,
)
```

### 3. 七种工作流模式

```python
from core.workflows import (
    intent_condition_workflow,           # 模式一：意图识别 + 条件路由
    team_supervisor_workflow,            # 模式二：Supervisor Python 版
    team_supervisor_graph_workflow,      # 模式三：Supervisor LangGraph 版
    mcp_react_workflow,                  # 模式四：MCP ReAct Agent（通用）
    mcp_supervisor_workflow,             # 模式五：Supervisor + MCP Agent（通用）
    capital_market_research_workflow,    # 模式六：资本市场研究助理（独立）
    capital_market_team_workflow,        # 模式七：Supervisor + 资本市场研究助理（团队协作）
)

# 模式一：先识别意图，再规则路由
result = await intent_condition_workflow("写个快排")

# 模式二：Supervisor 动态决策（串行链式调用，后 Agent 能看到前 Agent 输出）
result = await team_supervisor_workflow("查资料然后写代码", max_rounds=3)

# 模式三：LangGraph 版（支持可视化、断点续跑）
result = await team_supervisor_graph_workflow("查资料然后写代码", max_rounds=3)

# 模式四：MCP ReAct（单 Agent 调外部 Mock 工具）
result = await mcp_react_workflow("查询小米的股票代码")

# 模式五：MCP Supervisor（多 Agent + 外部工具协作）
result = await mcp_supervisor_workflow("搜索文档然后写代码")

# 模式六：资本市场研究助理（直连金融 MCP Server）
result = await capital_market_research_workflow(
    "查询腾讯最近一年的回购记录",
    server_url="http://127.0.0.1:1888/mcp/sse",
)

# 模式七：团队 Supervisor + 资本市场研究助理（自动路由金融/非金融问题）
result = await capital_market_team_workflow(
    "小米最近回购了多少股票？顺便告诉我天气怎么样",
    server_url="http://127.0.0.1:1888/mcp/sse",
)
```

---

## 扩展新 Agent

继承 `Agent` 基类，设置 `name` 和 `system_prompt` 即可：

```python
from core.agents.base import Agent
from core.llm.model_type import ModelType

class TranslationAgent(Agent):
    name = "translate"
    system_prompt = "你是一个翻译助手，把中文翻译成英文。"
    model_type = ModelType.GPT_4O_MINI

agent = TranslationAgent()
output = await agent.run("你好世界")
```

如果需要自定义逻辑，重写 `run` 方法：

```python
class LoggingAgent(Agent):
    async def run(self, query, context=None):
        print(f"[LOG] Query: {query}")
        return await super().run(query, context)
```

---

## 内置 LLM Provider

| Provider | 说明 | 环境变量 | 已注册模型 |
|----------|------|----------|-----------|
| OpenAI | 官方 OpenAI 模型 | `OPENAI_API_KEY` | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo |
| Bailian (百炼) | 阿里云百炼平台 | `DASHSCOPE_API_KEY` | qwen-max, qwen-plus, qwen-turbo, qwen3.5-plus, qwen3.5-flash |
| DeepSeek | DeepSeek 官方 | `DEEPSEEK_API_KEY` | deepseek-chat, deepseek-reasoner |
| **Kimi** | Moonshot 月之暗面 | `KIMI_API_KEY` | kimi-k2.5, kimi-k2, kimi-k1.5 |
| **OpenRouter** | 统一 API 网关 | `OPENROUTER_API_KEY` | anthropic/claude-3.5-sonnet, openai/gpt-4o, openai/gpt-4o-mini, deepseek/deepseek-chat |

Kimi 和 OpenRouter 已内置，在 `.env` 中填入对应 Key 即可直接使用。

---

## 扩展新 LLM Provider

以接入 **Anthropic Claude** 为例，只需三步：

### 1. 添加 provider 文件

新建 `llm/providers/anthropic.py`：

```python
from langchain_anthropic import ChatAnthropic
from llm.config import config

_cfg = config.providers.get("anthropic", {})
_api_key = _cfg.api_key if _cfg else ""

claude_3_5_sonnet = ChatAnthropic(
    model="claude-3-5-sonnet-20240620",
    api_key=_api_key,
    temperature=0.7,
)
```

### 2. 注册到工厂

编辑 `llm/model_type.py`，添加枚举值：

```python
class ModelType(str, Enum):
    # ... 已有模型
    CLAUDE_3_5_SONNET = "anthropic-claude-3-5-sonnet"
```

编辑 `llm/factory.py`，在 `__init__` 中导入并注册：

```python
from llm.providers import anthropic

self._models[ModelType.CLAUDE_3_5_SONNET] = anthropic.claude_3_5_sonnet
```

### 3. 配置 API Key

在 `.env` 中添加：

```bash
ANTHROPIC_API_KEY=sk-ant-xxx
```

或在 `config.yaml` 中添加：

```yaml
providers:
  anthropic:
    api_key: ${ANTHROPIC_API_KEY}
```

> **OpenRouter 特殊配置**：支持可选的 `app_name` 和 `app_url` 字段，用于在 OpenRouter 排行榜中标识你的应用。在 `config.yaml` 中配置即可自动传入 `X-Title` 和 `HTTP-Referer` 请求头。

---

## 项目结构

```
├── docker-compose.yml          # Elasticsearch + Kibana + MCP Server 部署
├── config.yaml                 # 配置文件（可选）
├── .env                        # 环境变量文件（可选，自动加载）
├── main.py                     # 主入口（hk_law / mcp 子命令）
├── Dockerfile.mcp              # MCP Server Docker 镜像
├── requirements.txt
├── core/                       # 底层多 Agent 框架
│   ├── agents/
│   │   ├── base.py             # Agent 统一基座（直接 LLM / ReAct + ToolProvider）
│   │   ├── toolkit.py          # ToolProvider ABC + MCPClientProvider
│   │   ├── specialized.py      # 预置子 Agent（search / code / chat / analysis）
│   │   └── mcp_agent.py        # MCP-enabled Agent（向后兼容包装器）
│   ├── llm/
│   │   ├── config.py           # 配置加载（支持 .env / config.yaml）
│   │   ├── model_type.py       # 模型枚举
│   │   ├── factory.py          # LLM 工厂
│   │   └── providers/          # 各提供商模型实例
│   ├── routing/
│   │   ├── intent.py           # 意图识别
│   │   ├── condition.py        # 条件路由
│   │   ├── supervisor.py       # Supervisor 循环版（串行上下文传递）
│   │   └── supervisor_graph.py # Supervisor LangGraph 版（串行上下文传递）
│   ├── utils/
│   │   └── logger.py           # 统一日志（stderr 输出，带颜色）
│   └── workflows.py            # 七种工作流组合
├── mcp_bridge/                 # MCP 协议集成（新增）
│   └── server/
│       └── demo_server.py      # FastMCP Demo Server（stdio / sse）
└── domains/                    # 业务域聚合（与 core/ 对齐）
    ├── hk_law/                 # 香港法律业务应用
    │   ├── main.py             # 法律系统入口（HKLawSystem）
    │   ├── agents/
    │   │   ├── __init__.py     # 法域配置表 + 工厂（生成 5 个法域 Agent）
    │   │   └── base.py         # 法律 Agent 基类（RAG 检索 + LLM 生成）
    │   ├── rag/
    │   │   ├── engine.py       # ES + 百炼 Embedding/Rerank RAG 引擎
    │   │   ├── ingest.py       # 文档索引 CLI 工具
    │   │   └── download.py     # 法律文档下载工具
    │   └── documents/          # 法律文档目录
    │       ├── criminal/
    │       ├── civil/
    │       ├── company/
    │       ├── employment/
    │       └── property/
    └── capital_market/         # 香港资本市场业务应用
        ├── main.py             # 业务入口（独立/团队两种工作流 + demo）
        └── agents/
            ├── __init__.py     # 导出 CapitalMarketAgent
            └── base.py         # CapitalMarketAgent（Agent + MCPClientProvider）
```

---

## RAG 技术栈

| 组件 | 选型 | 说明 |
|------|------|------|
| 向量数据库 | **Elasticsearch 8.15** | dense_vector + cosine similarity，支持 kNN 搜索 |
| Embedding | **百炼 text-embedding-v3** (云端) / **Ollama nomic-embed-text** (本地) | 1536 维 / 768 维，均支持中英双语 |
| Rerank | **百炼 gte-rerank** (云端) / **本地余弦相似度** (fallback) | 对 ES 召回结果重排序 |
| 文本切分 | RecursiveCharacterTextSplitter | chunk_size=800, overlap=100 |

Provider 通过环境变量切换：
- `EMBEDDING_PROVIDER=bailian\|ollama`
- `RERANK_PROVIDER=bailian\|local`

---

## 依赖

- Python >= 3.10
- LangGraph >= 0.3.0
- LangChain >= 0.3.0
- Pydantic >= 2.0.0
- Elasticsearch >= 8.15.0
- mcp >= 1.0.0
- fastmcp >= 2.0.0
- Docker & Docker Compose（用于部署 ES）
- Ollama（可选，用于本地 Embedding）

---

## License

MIT
