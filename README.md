# Multi-Agent + Intent Recognition Demo

一个轻量级的多 Agent 调用与意图识别示例项目，参考 [agenthub-py](https://github.com/agenthub-py) 的核心设计，简化为纯 Python 实现。

支持三种多 Agent 协作模式：
- **意图识别 + 条件路由**：LLM 识别意图，规则路由到固定 Agent
- **Team Supervisor（循环版）**：LLM 动态决策调用哪个 Agent，支持多轮
- **Team Supervisor（LangGraph 版）**：同上，但基于 LangGraph `Command(goto=...)` 实现

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

## 快速开始（通用 Demo）

```bash
python main.py
```

输出包含 6 个演示场景：
1. 意图识别 + 条件路由
2. Team Supervisor（循环版）
3. Team Supervisor（LangGraph 版）
4. 混合模型（不同 Agent 用不同 LLM）
5. 工厂 API 与动态注册
6. 扩展自定义 Agent

---

## 香港法律多 Agent 系统

本项目包含一个完整的 **香港法律多 Agent 系统**，每个法域对应独立的 Agent，基于 Elasticsearch 提供 RAG 能力。Embedding 和 Rerank 默认使用百炼云端模型，也支持切换到 Ollama 本地模型（零 API Key 成本）。

### 1. 部署 Elasticsearch

```bash
docker-compose up -d
```

启动后：
- Elasticsearch: http://localhost:9200
- Kibana: http://localhost:5601

### 2. 下载法律文档

从香港律政司 [电子版香港法例](https://www.elegislation.gov.hk/) 下载以下条例的 PDF/Word，放入对应目录：

| 法域 | 目录 | 建议下载条例 |
|------|------|-------------|
| 刑事 | `hk_law/documents/criminal/` | 《刑事罪行条例》(Cap. 200)、《盗窃罪条例》(Cap. 210) |
| 民事 | `hk_law/documents/civil/` | 《合约(第三方权利)条例》(Cap. 623)、《失实陈述条例》(Cap. 284) |
| 公司 | `hk_law/documents/company/` | 《公司条例》(Cap. 622) |
| 雇佣 | `hk_law/documents/employment/` | 《雇佣条例》(Cap. 57) |
| 物业 | `hk_law/documents/property/` | 《物业转易及财产条例》(Cap. 219)、《建筑物管理条例》(Cap. 344) |

下载地址：https://www.elegislation.gov.hk/

### 3. 构建向量索引

```bash
# 索引单个法域
python -m hk_law.rag.ingest criminal

# 索引所有法域
python -m hk_law.rag.ingest --all

# 重建索引（先删除再重建）
python -m hk_law.rag.ingest criminal --rebuild
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
python -m hk_law.rag.ingest --all
```

### 4. 运行法律问答

```bash
python -m hk_law.main
```

或使用 API：

```python
from hk_law.main import HKLawSystem

system = HKLawSystem()

# 意图识别 + 条件路由（适合单法域问题）
result = await system.ask("我被公司无故解雇了", mode="intent")

# Supervisor 动态协调（适合交叉法域问题）
result = await system.ask("我被公司解雇还欠工资", mode="supervisor")

print(result["output"])
```

---

## 核心用法

### 1. 指定模型类型

通过 `ModelType` 枚举选择 LLM，API Key 自动从配置读取：

```python
from llm.model_type import ModelType
from agents.specialized import CodeAgent

agent = CodeAgent(model_type=ModelType.QWEN_MAX)
output = await agent.run("写个 Python 快排")
```

### 2. 混合模型（每个 Agent 不同 LLM）

```python
from llm.model_type import ModelType
from agents.specialized import CodeAgent, ChatAgent
from workflows import team_supervisor_graph_workflow

code_agent = CodeAgent(model_type=ModelType.QWEN_MAX)
chat_agent = ChatAgent(model_type=ModelType.GPT_4O_MINI)

result = await team_supervisor_graph_workflow(
    "帮我写个快排，再闲聊两句",
    agents=[code_agent, chat_agent],
    supervisor_model=ModelType.GPT_4O,
)
```

### 3. 三种工作流模式

```python
from workflows import (
    intent_condition_workflow,      # 意图识别 + 条件路由
    team_supervisor_workflow,       # Supervisor Python 版
    team_supervisor_graph_workflow, # Supervisor LangGraph 版
)

# 模式一：先识别意图，再规则路由
result = await intent_condition_workflow("写个快排")

# 模式二：Supervisor 动态决策
result = await team_supervisor_workflow("查资料然后写代码", max_rounds=3)

# 模式三：LangGraph 版（支持可视化、断点续跑）
result = await team_supervisor_graph_workflow("查资料然后写代码", max_rounds=3)
```

---

## 扩展新 Agent

继承 `Agent` 基类，设置 `name` 和 `system_prompt` 即可：

```python
from agents.base import Agent
from llm.model_type import ModelType

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
├── docker-compose.yml          # Elasticsearch + Kibana 部署
├── config.yaml                 # 配置文件（可选）
├── .env                        # 环境变量文件（可选，自动加载）
├── main.py                     # 通用 demo 入口
├── requirements.txt
├── core/                       # 底层多 Agent 框架
│   ├── agents/
│   │   ├── base.py             # Agent 基类
│   │   └── specialized.py      # 预置子 Agent
│   ├── llm/
│   │   ├── config.py           # 配置加载（支持 .env / config.yaml）
│   │   ├── model_type.py       # 模型枚举
│   │   ├── factory.py          # LLM 工厂
│   │   └── providers/
│   │       ├── openai.py       # OpenAI 模型实例
│   │       ├── bailian.py      # 百炼 Chat 模型实例
│   │       ├── bailian_embedding.py  # 百炼 Embedding (text-embedding-v3)
│   │       ├── bailian_rerank.py     # 百炼 Rerank (gte-rerank)
│   │       ├── deepseek.py     # DeepSeek 模型实例
│   │       ├── kimi.py         # Kimi 模型实例
│   │       └── openrouter.py   # OpenRouter 模型实例
│   ├── routing/
│   │   ├── intent.py           # 意图识别
│   │   ├── condition.py        # 条件路由
│   │   ├── supervisor.py       # Supervisor 循环版
│   │   └── supervisor_graph.py # Supervisor LangGraph 版
│   └── workflows.py            # 三种工作流组合
└── hk_law/                     # 香港法律业务应用
    ├── main.py                 # 法律系统入口
    ├── agents/
    │   ├── base.py             # 法律 Agent 基类（继承 core.agents）
    │   ├── criminal.py         # 刑事法 Agent
    │   ├── civil.py            # 民事/合约法 Agent
    │   ├── company.py          # 公司法 Agent
    │   ├── employment.py       # 雇佣法 Agent
    │   └── property.py         # 物业法 Agent
    ├── rag/
    │   ├── engine.py           # ES + 百炼 Embedding/Rerank RAG 引擎
    │   └── ingest.py           # 文档索引 CLI 工具
    └── documents/              # 法律文档目录（需自行下载）
        ├── criminal/
        ├── civil/
        ├── company/
        ├── employment/
        └── property/
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
- Docker & Docker Compose（用于部署 ES）
- Ollama（可选，用于本地 Embedding）

---

## License

MIT
