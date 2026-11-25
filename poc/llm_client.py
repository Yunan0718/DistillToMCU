"""
DistillToMCU Phase 0b — LLM 客户端 (v2: 多 LLM 后端支持)
=========================================================
支持 DeepSeek V4 Flash + GPT-4o-mini 的 Cross-LLM 验证。
API Key 优先级：环境变量 > config.py

Cross-LLM 设计：
  - 同一批传感器序列分别发给 DeepSeek 和 GPT-4o-mini
  - 比较两者的 tool_call 决策一致性
  - 验证规则蒸馏是 LLM-agnostic 的
"""

import json
import os
import time
import requests
from config import (
    LLM_MODEL, LLM_BASE_URL, LLM_API_KEY, LLM_MAX_TOKENS,
    LLM_TEMPERATURE_AGENT, LLM_TEMPERATURE_RESIDENT, LLM_TEMPERATURE_CHECK,
    ACTUATORS, SENSORS,
)

# API Keys
_DS_API_KEY = os.environ.get("DEEPSEEK_API_KEY") or LLM_API_KEY
_OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY") or ""
_DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY") or ""

# v10.5: active backend switch (set by experiment scripts via --backend)
_ACTIVE_BACKEND = os.environ.get("LLM_BACKEND", "deepseek-v4-flash")

# Cross-LLM 配置
CROSS_LLM_CONFIGS = {
    "deepseek-v4-flash": {
        "base_url": "https://api.deepseek.com",
        "api_key": lambda: _DS_API_KEY,
        "model": "deepseek-v4-flash",
        "temperature": LLM_TEMPERATURE_AGENT,
        "headers_extra": {},
    },
    "gpt-4o-mini": {
        "base_url": "https://api.openai.com",
        "api_key": lambda: _OPENAI_API_KEY,
        "model": "gpt-4o-mini",
        "temperature": LLM_TEMPERATURE_AGENT,
        "headers_extra": {},
    },
    "qwen3.7-flash-2026-07-15": {
        # NOTE: call_llm_with_backend appends "/v1/chat/completions",
        # so base_url must NOT include /v1 (404 otherwise).
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode",
        "api_key": lambda: _DASHSCOPE_API_KEY,
        "model": "qwen3.7-flash-2026-07-15",
        "temperature": LLM_TEMPERATURE_AGENT,
        "headers_extra": {},
    },
}


def get_available_llms() -> list[str]:
    """返回当前可用的 LLM 列表（根据环境变量中的 API Key）"""
    available = []
    if len(_DS_API_KEY) > 10:
        available.append("deepseek-v4-flash")
    if len(_OPENAI_API_KEY) > 10:
        available.append("gpt-4o-mini")
    return available


def call_llm_with_backend(
    messages: list,
    tools: list = None,
    backend: str = "deepseek-v4-flash",
    temperature: float = 0.0,
    max_tokens: int | None = None,
) -> dict:
    """
    用指定 LLM 后端调用 API。支持 DeepSeek 和 OpenAI 兼容接口。

    Returns:
        {content, tool_calls, finish_reason, latency_ms, model}
    """
    if backend not in CROSS_LLM_CONFIGS:
        return _mock_llm_response(messages, tools)

    cfg = CROSS_LLM_CONFIGS[backend]
    api_key = cfg["api_key"]()

    if len(api_key) < 10:
        return _mock_llm_response(messages, tools)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        **cfg["headers_extra"],
    }
    payload = {
        "model": cfg["model"],
        "messages": messages,
        "max_tokens": max_tokens or LLM_MAX_TOKENS,
        "temperature": temperature or cfg["temperature"],
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    t0 = time.time()
    try:
        resp = requests.post(
            f'{cfg["base_url"]}/v1/chat/completions',
            headers=headers, json=payload, timeout=30,
        )
    except requests.RequestException as e:
        return {
            "content": None, "tool_calls": None,
            "finish_reason": "error", "latency_ms": 0,
            "model": backend, "error": str(e),
        }

    latency_ms = int((time.time() - t0) * 1000)

    if resp.status_code != 200:
        return {
            "content": None, "tool_calls": None,
            "finish_reason": f"error_{resp.status_code}",
            "latency_ms": latency_ms, "model": backend,
            "error": resp.text[:200],
        }

    body = resp.json()
    choice = body["choices"][0]
    msg = choice["message"]

    return {
        "content": msg.get("content"),
        "tool_calls": msg.get("tool_calls"),
        "finish_reason": choice.get("finish_reason", "stop"),
        "latency_ms": latency_ms,
        "model": body.get("model", backend),
    }


# ========== Tool Definitions ==========

def build_tools():
    """构建 Cloud Agent 可用的 tool 列表"""
    tools = []
    for name, info in ACTUATORS.items():
        props = {
            "command": {
                "type": "string",
                "enum": ["on", "off", "set"],
                "description": f"Action to perform on {name}"
            }
        }
        for param in info["params"]:
            props[param] = {"type": "number", "description": f"{param} value for {name}"}

        tools.append({
            "type": "function",
            "function": {
                "name": f"{name}_control",
                "description": f"Control the {name} device ({info['type']}). "
                               f"Safety level: {info['safety']} (0=query, 1=comfort, 2=high-energy, 3=safety-critical)",
                "parameters": {
                    "type": "object",
                    "properties": props,
                    "required": ["command"],
                }
            }
        })

    # 添加只读查询 tools
    tools.append({
        "type": "function",
        "function": {
            "name": "read_sensors",
            "description": f"Read current values from all sensors: {', '.join(SENSORS)}",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            }
        }
    })

    return tools


# ========== LLM API 调用 ==========

def _call_llm(messages, tools=None, tool_choice="auto", temperature=0.0, stream=False):
    """底层 LLM API 调用（默认用 DeepSeek）"""
    return call_llm_with_backend(messages, tools, backend=_ACTIVE_BACKEND,
                                 temperature=temperature)


# ========== 三种 LLM 角色 ==========

def cloud_agent_think(system_prompt, user_input, sensors, history_summary=""):
    """
    云端 Agent：根据用户指令 + 传感器状态，做出结构化决策。
    返回 tool_call(s) 或纯文本回复。
    """
    tools = build_tools()

    sensor_text = "\n".join([f"  {k}: {v}" for k, v in sensors.items()
                             if v is not None])

    messages = [
        {"role": "system", "content": system_prompt},
    ]
    if history_summary:
        messages.append({"role": "system", "content": f"Recent context:\n{history_summary}"})

    messages.append({
        "role": "user",
        "content": f"Current sensors:\n{sensor_text}\n\nUser says: \"{user_input}\"\n\n"
                   f"Decide what action to take. Use the available tools if device control is needed."
    })

    return _call_llm(messages, tools=tools, temperature=LLM_TEMPERATURE_AGENT)


def resident_simulate(sensors, time_context=""):
    """
    住户模拟器：根据传感器状态，生成自然语言的 user_input。
    模拟的是真实用户在这个传感器环境下可能说的话。
    """
    sensor_text = "\n".join([f"  {k}: {v}" for k, v in sensors.items()])

    messages = [{
        "role": "system",
        "content": (
            "You are a resident in a smart home. Given the current sensor readings "
            "and time context, generate a natural language command or question "
            "that a real person might say in this situation.\n\n"
            "Rules:\n"
            "- Be natural and varied. Different people say things differently.\n"
            "- Sometimes you just want information ('What's the temperature?').\n"
            "- Sometimes you want action ('Turn on the light, it's dark').\n"
            "- Sometimes you express discomfort ('It's too hot in here').\n"
            "- Keep it short (5-15 words).\n"
            "- Do NOT include the sensor values in your speech "
            "(say 'it's hot' not 'the temperature is 31.2').\n\n"
            "Return ONLY the utterance, no explanation."
        )
    }]
    if time_context:
        messages.append({"role": "user", "content": time_context})
    messages.append({
        "role": "user",
        "content": f"Sensors: {sensor_text}\n\nWhat would a resident say right now? Say ONLY the utterance:"
    })

    return _call_llm(messages, temperature=LLM_TEMPERATURE_RESIDENT)


def sanity_check_rule(rule_text):
    """
    规则 sanity check：让 LLM 判断一条候选规则是否合理。
    """
    messages = [{
        "role": "system",
        "content": (
            "You are a safety validator for smart home automation rules. "
            "Given a candidate rule, evaluate whether it is reasonable, safe, "
            "and free of obvious edge-case problems.\n\n"
            "Return a JSON object:\n"
            '{"reasonable": true/false, "reason": "one sentence explanation"}'
        )
    }, {
        "role": "user",
        "content": f"Candidate rule:\n{rule_text}\n\nIs this rule reasonable?"
    }]

    result = _call_llm(messages, temperature=LLM_TEMPERATURE_CHECK)
    try:
        if not result or not result.get("content"):
            # v7 修复：安全检查默认拒绝。安全门的正确默认是拒，不是放。
            return {"reasonable": False,
                    "reason": "LLM check skipped (empty response), defaulting to REJECT for safety"}
        content = result.get("content", "{}")
        parsed = json.loads(content)
        if not isinstance(parsed, dict) or "reasonable" not in parsed:
            return {"reasonable": False,
                    "reason": "LLM output missing 'reasonable' key, defaulting to REJECT"}
        return parsed
    except (json.JSONDecodeError, TypeError):
        return {"reasonable": False,
                "reason": "could not parse LLM output, defaulting to REJECT for safety"}


# ========== Mock（无 API Key 时的降级） ==========

def _mock_llm_response(messages, tools):
    """无 API Key 时的 mock 返回，用于测试代码框架"""
    import random
    import time

    latency_ms = random.randint(800, 2000)

    # 简单规则：根据 sensor 值生成 mock tool_call
    user_msg = messages[-1].get("content", "") if messages else ""

    if "temperature" in user_msg.lower() or "hot" in user_msg.lower() or "热" in user_msg.lower():
        return {
            "content": None,
            "tool_calls": [{
                "id": "mock_001",
                "type": "function",
                "function": {
                    "name": "fan_control",
                    "arguments": json.dumps({"command": "on", "speed": 2})
                }
            }],
            "finish_reason": "tool_calls",
            "latency_ms": latency_ms,
            "model": "mock",
        }
    elif "light" in user_msg.lower() or "dark" in user_msg.lower() or "暗" in user_msg.lower() or "灯" in user_msg.lower():
        return {
            "content": None,
            "tool_calls": [{
                "id": "mock_002",
                "type": "function",
                "function": {
                    "name": "led_control",
                    "arguments": json.dumps({"command": "on", "brightness": 60})
                }
            }],
            "finish_reason": "tool_calls",
            "latency_ms": latency_ms,
            "model": "mock",
        }
    elif "morning" in user_msg.lower() or "起床" in user_msg.lower() or "早" in user_msg.lower():
        return {
            "content": None,
            "tool_calls": [{
                "id": "mock_003",
                "type": "function",
                "function": {
                    "name": "curtain_control",
                    "arguments": json.dumps({"command": "on", "position": 100})
                }
            }],
            "finish_reason": "tool_calls",
            "latency_ms": latency_ms,
            "model": "mock",
        }
    else:
        return {
            "content": "OK, I've noted your request. No device action needed.",
            "tool_calls": None,
            "finish_reason": "stop",
            "latency_ms": latency_ms,
            "model": "mock",
        }
