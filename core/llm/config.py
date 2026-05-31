"""
LLM 配置加载器
支持从 config.yaml 读取，自动做环境变量替换 (${ENV_VAR})。
也支持从 .env 文件加载环境变量。
未找到配置文件时，fallback 到直接读取环境变量。
"""
import os
import re
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field

# 自动加载 .env 文件（如果存在）
try:
    from dotenv import load_dotenv

    _env_files = [".env", ".env.local", ".env.development"]
    for _env_file in _env_files:
        if Path(_env_file).exists():
            load_dotenv(_env_file, override=False)
            break
except ImportError:
    pass


class ProviderConfig(BaseModel):
    """单个 LLM Provider 的配置"""

    api_key: str = ""
    base_url: str = ""


class LLMConfig(BaseModel):
    """LLM 全局配置"""

    providers: Dict[str, ProviderConfig] = Field(default_factory=dict)


def _replace_env_vars(value: Any) -> Any:
    """递归替换字符串中的 ${ENV_VAR} 为对应环境变量值"""
    if isinstance(value, str):
        pattern = re.compile(r"\$\{([^}]+)\}")

        def replacer(match):
            env_var = match.group(1)
            return os.getenv(env_var, match.group(0))

        return pattern.sub(replacer, value)
    if isinstance(value, dict):
        return {k: _replace_env_vars(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_replace_env_vars(v) for v in value]
    return value


def load_config(config_path: str = None) -> LLMConfig:
    """
    加载 LLM 配置。
    查找顺序:
      1. 传入的 config_path
      2. 当前目录下的 config.yaml
      3. 当前目录下的 config.json
      4. 环境变量直接读取（fallback）
    """
    # 确定配置文件路径
    if config_path is None:
        candidates = ["config.yaml", "config.yml", "config.json"]
        for c in candidates:
            if Path(c).exists():
                config_path = c
                break

    raw_data: Dict[str, Any] = {"providers": {}}

    if config_path and Path(config_path).exists():
        if config_path.endswith(".json"):
            import json

            with open(config_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
        else:
            try:
                import yaml

                with open(config_path, "r", encoding="utf-8") as f:
                    raw_data = yaml.safe_load(f) or {}
            except ImportError:
                raise ImportError(
                    "PyYAML is required to load .yaml config. "
                    "Install it: pip install pyyaml, or use config.json instead."
                )

    # 环境变量替换
    raw_data = _replace_env_vars(raw_data)

    # 如果配置文件没提供某些 provider，尝试从环境变量补全
    providers = raw_data.get("providers", {})
    _env_map = {
        "openai": "OPENAI_API_KEY",
        "bailian": "DASHSCOPE_API_KEY",  # 百炼使用 DASHSCOPE 前缀
        "deepseek": "DEEPSEEK_API_KEY",
        "kimi": "KIMI_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
    }
    for provider_name, env_var in _env_map.items():
        if provider_name not in providers:
            providers[provider_name] = {}
        env_key = os.getenv(env_var, "")
        if env_key and not providers[provider_name].get("api_key"):
            providers[provider_name]["api_key"] = env_key

    raw_data["providers"] = providers
    return LLMConfig(**raw_data)


# 全局单例（首次导入时加载）
config = load_config()
