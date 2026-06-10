"""Часть 1: синхронный LLM-клиент поверх сырого `requests` + ручной tool calling.

В этой части нельзя пользоваться библиотекой `openai` и нативным параметром
`tools=` / function-calling API. Формат вызова инструментов вы описываете в
system prompt сами (Qwen-style: блоки `<tool_call>{...}</tool_call>`), а парсинг
и исполнение делает ваш код (харнесс).

Реализуйте всё, помеченное `raise NotImplementedError`. Менять публичные
сигнатуры не рекомендуется — на них завязаны тесты.
"""
from __future__ import annotations

import json
import re
from typing import Any, Callable

import requests

# Регулярка для вытаскивания tool call из текста модели (Qwen-style). Подсказка —
# можете пользоваться как есть.
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


class LLMClient:
    """Синхронный клиент OpenAI-совместимого Chat Completions API поверх requests.

    Никакого `openai` и `tools=` — только сырой HTTP POST на
    `{base_url}/chat/completions`.

    HTTP-слой (`_headers` / `_payload` / `chat` + учёт токенов) уже реализован —
    чтобы вы сразу занялись сутью Части 1: формированием system prompt, парсингом
    и исполнением tool call'ов (функции ниже всё ещё `raise NotImplementedError`).
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.total_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        }
        self.n_requests = 0

    def _headers(self) -> dict[str, str]:
        """Заголовки запроса (авторизация Bearer + JSON)."""
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _payload(self, messages: list[dict]) -> dict[str, Any]:
        """Тело запроса к chat/completions (model, messages, temperature, ...)."""
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }

    def chat(self, messages: list[dict]) -> str:
        """Отправить messages модели и вернуть текст ответа ассистента."""
        url = f"{self.base_url}/chat/completions"
        resp = requests.post(
            url,
            headers=self._headers(),
            json=self._payload(messages),
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage") or {}
        for key in self.total_usage:
            self.total_usage[key] += usage.get(key, 0) or 0
        self.n_requests += 1

        return data["choices"][0]["message"]["content"] or ""


def render_tools_system_prompt(tool_schemas: list[dict]) -> str:
    """Собрать system prompt со списком инструментов (Qwen-style формат вызова)."""
    tools_json = json.dumps(tool_schemas, ensure_ascii=False, indent=2)
    return (
        "Ты — ассистент с доступом к инструментам.\n\n"
        "# Tools\n"
        "Ты можешь вызвать один или несколько инструментов, если это нужно для ответа.\n"
        "Доступные инструменты описаны JSON-схемами ниже:\n"
        "<tools>\n"
        f"{tools_json}\n"
        "</tools>\n\n"
        "Чтобы вызвать инструмент, верни JSON-объект с именем инструмента и "
        "аргументами внутри XML-тегов <tool_call></tool_call>. "
        "Используй ровно такой формат:\n"
        "<tool_call>\n"
        '{"name": "tool_name", "arguments": {}}\n'
        "</tool_call>\n\n"
        "Поле `arguments` всегда должно быть JSON-объектом. "
        "Если нужно вызвать несколько инструментов, верни несколько блоков "
        "<tool_call>...</tool_call> подряд.\n"
        "Если инструмент не нужен, ответь обычным текстом без блока <tool_call>."
    )


def parse_tool_calls(text: str) -> list[dict]:
    """Найти все <tool_call>{...}</tool_call> в тексте модели и распарсить JSON."""
    calls: list[dict] = []
    for match in TOOL_CALL_RE.finditer(text):
        raw_json = match.group(1)
        try:
            parsed = json.loads(raw_json)
        except json.JSONDecodeError:
            continue

        if isinstance(parsed, dict):
            calls.append(parsed)

    return calls


def execute_tool_calls(
    calls: list[dict],
    tools: dict[str, Callable[..., Any]],
) -> list[str]:
    """Исполнить распарсенные tool call'ы и вернуть текстовые результаты."""
    results: list[str] = []

    for call in calls:
        if not isinstance(call, dict):
            results.append("Ошибка: tool call должен быть JSON-объектом")
            continue

        name = call.get("name")
        arguments = call.get("arguments", {})

        if arguments is None:
            arguments = {}

        if not isinstance(name, str) or not name:
            results.append("Ошибка: в tool call отсутствует корректное поле `name`")
            continue

        if name not in tools:
            results.append(f"Ошибка: неизвестный инструмент: {name}")
            continue

        if not isinstance(arguments, dict):
            results.append(
                f"Ошибка: arguments для инструмента `{name}` должен быть JSON-объектом"
            )
            continue

        try:
            result = tools[name](**arguments)
        except TypeError as exc:
            results.append(f"Ошибка аргументов инструмента `{name}`: {exc}")
            continue
        except Exception as exc:
            results.append(
                f"Ошибка при выполнении инструмента `{name}` "
                f"({type(exc).__name__}): {exc}"
            )
            continue

        if isinstance(result, str):
            results.append(result)
        else:
            try:
                results.append(json.dumps(result, ensure_ascii=False))
            except TypeError:
                results.append(str(result))

    return results


def wrap_tool_response(result: str) -> dict:
    """Обернуть результат инструмента в сообщение для следующего хода модели."""
    return {
        "role": "user",
        "content": f"<tool_response>\n{result}\n</tool_response>",
    }


def run_tool_loop(
    client: LLMClient,
    user_msg: str,
    tools: dict[str, Callable[..., Any]],
    tool_schemas: list[dict],
    max_steps: int = 6,
    system_extra: str = "",
    messages: list[dict] | None = None,
    verbose: bool = False,
) -> str:
    """Цикл tool calling: модель -> парсинг -> исполнение -> ответ -> повтор."""
    system_prompt = render_tools_system_prompt(tool_schemas)
    if system_extra:
        system_prompt = f"{system_prompt}\n\n{system_extra}"

    if messages is None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
    else:
        has_system = any(message.get("role") == "system" for message in messages)
        if not has_system:
            messages.insert(0, {"role": "system", "content": system_prompt})
        if user_msg:
            messages.append({"role": "user", "content": user_msg})

    for step in range(max_steps):
        reply = client.chat(messages)

        if verbose:
            print(f"[assistant step {step + 1}] {reply}")

        messages.append({"role": "assistant", "content": reply})
        calls = parse_tool_calls(reply)

        if not calls:
            return reply

        results = execute_tool_calls(calls, tools)
        for result in results:
            if verbose:
                print(f"[tool response] {result}")
            messages.append(wrap_tool_response(result))

    return f"Ошибка: достигнут лимит шагов ({max_steps}) без финального ответа"
