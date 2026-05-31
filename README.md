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
OPENAI_API_KEY=sk-xxx
DASHSCOPE_API_KEY=sk-yyy
DEEPSEEK_API_KEY=sk-zzz
KIMI_API_KEY=sk-aaa
OPENROUTER_API_KEY=sk-bbb
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

## 扩展新 LLM Provider

以接入 **Kimi** 和 **OpenRouter** 为例（项目已内置）：

### 1. 添加 provider 文件

新建 `llm/providers/kimi.py`：

```python
from langchain_openai import ChatOpenAI
from llm.config import config

_cfg = config.providers.get("kimi", {})
_api_key = _cfg.api_key if _cfg else ""
_base_url = _cfg.base_url if _cfg else "https://api.moonshot.cn/v1"

kimi_k2_5 = ChatOpenAI(
    model="kimi-k2.5",
    api_key=_api_key,
    base_url=_base_url,
    temperature=0.7,
)
```

新建 `llm/providers/openrouter.py`：

```python
from langchain_openai import ChatOpenAI
from llm.config import config

_cfg = config.providers.get("openrouter", {})
_api_key = _cfg.api_key if _cfg else ""
_base_url = _cfg.base_url if _cfg else "https://openrouter.ai/api/v1"

# OpenRouter 推荐传入应用标识头（可选）
_default_headers = {}
if _cfg and getattr(_cfg, "app_name", None):
    _default_headers["X-Title"] = _cfg.app_name
    _default_headers["HTTP-Referer"] = getattr(_cfg, "app_url", "")

openrouter_claude_3_5_sonnet = ChatOpenAI(
    model="anthropic/claude-3.5-sonnet",
    api_key=_api_key,
    base_url=_base_url,
    temperature=0.7,
    default_headers=_default_headers,
)
```

### 2. 注册到工厂

编辑 `llm/model_type.py`，添加枚举值：

```python
class ModelType(str, Enum):
    # ... 已有模型
    KIMI_K2_5 = "kimi-kimi-k2.5"
    OR_CLAUDE_3_5_SONNET = "openrouter-claude-3.5-sonnet"
```

编辑 `llm/factory.py`，在 `__init__` 中注册：

```python
from llm.providers import kimi, openrouter

self._models[ModelType.KIMI_K2_5] = kimi.kimi_k2_5
self._models[ModelType.OR_CLAUDE_3_5_SONNET] = openrouter.openrouter_claude_3_5_sonnet
```

### 3. 配置 API Key

在 `.env` 中添加：

```bash
KIMI_API_KEY=sk-xxx
OPENROUTER_API_KEY=sk-or-xxx
```

或在 `config.yaml` 中添加：

```yaml
providers:
  kimi:
    api_key: ${KIMI_API_KEY}
    base_url: https://api.moonshot.cn/v1
  openrouter:
    api_key: ${OPENROUTER_API_KEY}
    base_url: https://openrouter.ai/api/v1
```

---

## 项目结构

```
├── config.yaml                 # 配置文件（可选）
├── .env                        # 环境变量文件（可选，自动加载）
├── main.py                     # 入口演示
├── workflows.py                # 三种工作流组合
├── requirements.txt
├── agents/
│   ├── base.py                 # Agent 基类
│   └── specialized.py          # 预置子 Agent
├── llm/
│   ├── config.py               # 配置加载（支持 .env / config.yaml）
│   ├── model_type.py           # 模型枚举
│   ├── factory.py              # LLM 工厂
│   └── providers/
│       ├── openai.py           # OpenAI 模型实例
│       ├── bailian.py          # 百炼模型实例
│       ├── deepseek.py         # DeepSeek 模型实例
│       ├── kimi.py             # Kimi 模型实例
│       └── openrouter.py       # OpenRouter 模型实例
└── routing/
    ├── intent.py               # 意图识别
    ├── condition.py            # 条件路由
    ├── supervisor.py           # Supervisor 循环版
    └── supervisor_graph.py     # Supervisor LangGraph 版
```

---

## 依赖

- Python >= 3.10
- LangGraph >= 0.3.0
- LangChain >= 0.3.0
- Pydantic >= 2.0.0

---

## License

MIT
