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
        # Накопленный расход токенов по всем вызовам chat() этого клиента.
        # API возвращает usage в каждом ответе — суммируем, чтобы было видно,
        # «сколько стоит» агент.
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
        """Отправить messages модели и вернуть текст ответа ассистента.

        Args:
            messages: список сообщений вида {"role": ..., "content": ...}.
                Допустимые роли — "system" / "user" / "assistant" (tool calling
                мы делаем вручную через текстовые теги, поэтому роль "tool" не
                используется — результат инструмента кладётся как user-сообщение).

        Returns:
            Текст из choices[0].message.content (пустая строка, если content == null).
        """
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
    """Собрать system prompt со списком инструментов (Qwen-style формат вызова).

    System prompt должен содержать: JSON-схемы инструментов и инструкцию, что для
    вызова инструмента нужно вернуть блок `<tool_call>{"name": ..., "arguments": ...}</tool_call>`,
    а без вызова — ответить обычным текстом.

    Args:
        tool_schemas: список JSON-схем инструментов (формат OpenAI function:
            {"type": "function", "function": {"name", "description", "parameters"}}).

    Returns:
        Готовый system prompt.
    """
    tools_json = json.dumps(tool_schemas, ensure_ascii=False, indent=2)
    return (
        "You are a helpful assistant that can use tools when needed.\n\n"
        "# Tools\n"
        "You may call one or more tools to help answer the user.\n"
        "The available tools are described by the following JSON schemas:\n"
        "<tools>\n"
        f"{tools_json}\n"
        "</tools>\n\n"
        "To call a tool, return a JSON object with the tool name and arguments "
        "inside <tool_call></tool_call> XML tags. Use exactly this format:\n"
        "<tool_call>\n"
        '{"name": "tool_name", "arguments": {}}\n'
        "</tool_call>\n\n"
        "The `arguments` value must be a JSON object. If you need multiple tool "
        "calls, return multiple <tool_call>...</tool_call> blocks.\n"
        "If no tool is needed, answer the user directly with ordinary text and "
        "do not include a <tool_call> block."
    )


def parse_tool_calls(text: str) -> list[dict]:
    """Найти все <tool_call>{...}</tool_call> в тексте модели и распарсить JSON.

    Args:
        text: ответ модели.

    Returns:
        Список dict вида {"name": ..., "arguments": {...}}. Блоки с невалидным
        JSON нужно пропускать (а не падать).
    """
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
    """Исполнить распарсенные tool call'ы и вернуть текстовые результаты.

    Args:
        calls: список вызовов из parse_tool_calls.
        tools: реестр инструментов {имя: функция}.

    Returns:
        Список строк-результатов (по одной на каждый вызов). Ошибки (неизвестный
        инструмент, неверные аргументы, исключение в инструменте) нужно возвращать
        как текст, а не пробрасывать — чтобы модель могла их прочитать и исправиться.
    """
    results: list[str] = []

    for call in calls:
        if not isinstance(call, dict):
            results.append("ERROR: tool call must be a JSON object")
            continue

        name = call.get("name")
        arguments = call.get("arguments", {})
        if arguments is None:
            arguments = {}

        if not isinstance(name, str) or not name:
            results.append("ERROR: tool call is missing a valid string field `name`")
            continue

        if name not in tools:
            results.append(f"ERROR: unknown tool: {name}")
            continue

        if not isinstance(arguments, dict):
            results.append(
                f"ERROR: arguments for tool `{name}` must be a JSON object"
            )
            continue

        try:
            result = tools[name](**arguments)
        except TypeError as exc:
            results.append(f"ERROR: invalid arguments for tool `{name}`: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - tool errors are returned to the model
            results.append(
                f"ERROR: tool `{name}` raised {type(exc).__name__}: {exc}"
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
    """Обернуть результат инструмента в сообщение для следующего хода модели.

    Tool calling делаем вручную, поэтому результат возвращаем как user-сообщение
    в обёртке <tool_response>...</tool_response> (так его ждёт chat template Qwen).
    """
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
    """Цикл tool calling: модель -> парсинг -> исполнение -> ответ -> повтор.

    Args:
        client: LLMClient (или любой объект с .chat(messages) -> str).
        user_msg: запрос пользователя.
        tools: реестр инструментов {имя: функция}.
        tool_schemas: JSON-схемы инструментов для system prompt.
        max_steps: ограничитель числа итераций (защита от зацикливания).
        system_extra: доп. инструкции, дописываются в конец system prompt.
        messages: если передан — продолжаем существующий диалог (мутируется на
            месте); иначе создаётся новый.
        verbose: печатать ли шаги.

    Returns:
        Финальный текстовый ответ модели (без tool_call) либо сообщение о лимите шагов.
    """
    system_prompt = render_tools_system_prompt(tool_schemas)
    if system_extra:
        system_prompt = f"{system_prompt}\n\n{system_extra}"

    if messages is None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg},
        ]
    else:
        # Продолжаем переданный диалог и мутируем его на месте, как обещано
        # в сигнатуре. Если системного сообщения нет, добавим его в начало.
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

    return f"ERROR: max_steps exceeded ({max_steps}) without final answer"
