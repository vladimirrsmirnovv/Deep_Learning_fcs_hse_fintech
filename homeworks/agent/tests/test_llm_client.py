"""Тесты Части 1: LLMClient (сырой requests) + ручной tool calling.

Все тесты детерминированы и работают против локального mock-сервера — реальная
модель/ключ/сеть не нужны.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))  # для mock_llm_server
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # для llm_client

from llm_client import (  # noqa: E402
    LLMClient,
    execute_tool_calls,
    parse_tool_calls,
    render_tools_system_prompt,
    run_tool_loop,
    wrap_tool_response,
)
from mock_llm_server import MockLLMServer, ScriptedLLM  # noqa: E402


# --- LLMClient.chat: корректный HTTP-запрос и разбор ответа ---

def test_chat_returns_assistant_text():
    with MockLLMServer(["привет из mock"]) as srv:
        client = LLMClient(base_url=srv.base_url, api_key="k", model="m")
        out = client.chat([{"role": "user", "content": "hi"}])
        assert out == "привет из mock"


def test_chat_request_shape():
    with MockLLMServer(["ok"]) as srv:
        client = LLMClient(base_url=srv.base_url, api_key="secret", model="my-model",
                           temperature=0.0, max_tokens=128)
        client.chat([{"role": "user", "content": "hello"}])
        req = srv.requests[-1]
        # endpoint
        assert req["path"].endswith("/chat/completions")
        # авторизация Bearer
        assert req["headers"].get("Authorization") == "Bearer secret"
        # тело
        body = req["payload"]
        assert body["model"] == "my-model"
        assert body["messages"] == [{"role": "user", "content": "hello"}]
        assert body["temperature"] == 0.0
        assert body["max_tokens"] == 128


def test_chat_null_content_is_empty_string():
    # content == null должен превратиться в "" (а не None/исключение).
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):  # noqa: ANN002
            pass

        def do_POST(self):  # noqa: N802
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            body = json.dumps({"choices": [{"message": {"content": None}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        client = LLMClient(base_url=f"http://127.0.0.1:{httpd.server_address[1]}/v1",
                           api_key="k", model="m")
        assert client.chat([{"role": "user", "content": "x"}]) == ""
    finally:
        httpd.shutdown()
        httpd.server_close()


# --- system prompt со списком инструментов ---

SCHEMAS = [
    {"type": "function", "function": {"name": "get_weather",
        "description": "погода", "parameters": {"type": "object",
        "properties": {"city": {"type": "string"}}, "required": ["city"]}}},
]


def test_render_system_prompt_contains_tools_and_format():
    prompt = render_tools_system_prompt(SCHEMAS)
    assert "get_weather" in prompt          # имя инструмента в промпте
    assert "<tool_call>" in prompt           # формат вызова описан
    assert "city" in prompt                  # схема аргументов попала в промпт


# --- парсинг tool call ---

def test_parse_single_tool_call():
    text = 'Сейчас проверю. <tool_call>{"name": "get_weather", "arguments": {"city": "Paris"}}</tool_call>'
    calls = parse_tool_calls(text)
    assert calls == [{"name": "get_weather", "arguments": {"city": "Paris"}}]


def test_parse_multiple_tool_calls():
    text = ('<tool_call>{"name": "a", "arguments": {}}</tool_call>\n'
            '<tool_call>{"name": "b", "arguments": {"x": 1}}</tool_call>')
    calls = parse_tool_calls(text)
    assert [c["name"] for c in calls] == ["a", "b"]
    assert calls[1]["arguments"] == {"x": 1}


def test_parse_ignores_invalid_json():
    text = '<tool_call>{это не json}</tool_call>'
    assert parse_tool_calls(text) == []


def test_parse_no_calls_in_plain_text():
    assert parse_tool_calls("просто ответ без вызовов") == []


# --- исполнение tool call ---

def _tools():
    return {
        "add": lambda a, b: a + b,
        "boom": lambda: (_ for _ in ()).throw(RuntimeError("ой")),
    }


def test_execute_known_tool():
    out = execute_tool_calls([{"name": "add", "arguments": {"a": 2, "b": 3}}], _tools())
    assert out == ["5"]


def test_execute_unknown_tool_returns_error_text():
    out = execute_tool_calls([{"name": "nope", "arguments": {}}], _tools())
    assert len(out) == 1 and "неизвестн" in out[0].lower()


def test_execute_bad_arguments_returns_error_text():
    out = execute_tool_calls([{"name": "add", "arguments": {"a": 1}}], _tools())
    assert len(out) == 1 and "ошибка" in out[0].lower()


def test_execute_tool_exception_is_caught():
    out = execute_tool_calls([{"name": "boom", "arguments": {}}], _tools())
    assert len(out) == 1 and "ой" in out[0]


def test_wrap_tool_response_format():
    msg = wrap_tool_response("результат")
    assert msg["role"] == "user"
    assert "<tool_response>" in msg["content"] and "результат" in msg["content"]


# --- полный цикл tool calling против mock-сервера ---

def test_run_tool_loop_executes_then_finishes():
    script = [
        '<tool_call>{"name": "add", "arguments": {"a": 2, "b": 2}}</tool_call>',
        "Готово: 2 + 2 = 4.",
    ]
    tools = {"add": lambda a, b: a + b}
    with MockLLMServer(script) as srv:
        client = LLMClient(base_url=srv.base_url, api_key="k", model="m")
        final = run_tool_loop(client, "сложи 2 и 2", tools, SCHEMAS, max_steps=5)
        srv.sync()
    assert final == "Готово: 2 + 2 = 4."
    # был ровно один tool call + финальный ответ => 2 обращения к модели
    assert srv.step == 2


def test_run_tool_loop_no_calls_returns_immediately():
    with MockLLMServer(["сразу ответ"]) as srv:
        client = LLMClient(base_url=srv.base_url, api_key="k", model="m")
        final = run_tool_loop(client, "вопрос", {}, SCHEMAS, max_steps=5)
    assert final == "сразу ответ"


def test_run_tool_loop_respects_max_steps():
    # модель всё время зовёт тул -> должен сработать лимит
    loop_call = '<tool_call>{"name": "noop", "arguments": {}}</tool_call>'
    with MockLLMServer([loop_call]) as srv:
        client = LLMClient(base_url=srv.base_url, api_key="k", model="m")
        final = run_tool_loop(client, "x", {"noop": lambda: "ok"}, SCHEMAS, max_steps=3)
    assert "лимит" in final.lower()


# --- ScriptedLLM как оффлайн-двойник клиента (для Части 2) ---

def test_scripted_llm_interface():
    llm = ScriptedLLM(["a", "b"])
    assert llm.chat([{"role": "user", "content": "1"}]) == "a"
    assert llm.chat([{"role": "user", "content": "2"}]) == "b"
    assert llm.chat([]) == "b"  # последний повторяется
    assert len(llm.calls) == 3


# --- запрет нативного tool calling: tools= не должен уходить в тело запроса ---

def test_client_does_not_send_native_tools_field():
    with MockLLMServer(["ok"]) as srv:
        client = LLMClient(base_url=srv.base_url, api_key="k", model="m")
        client.chat([{"role": "user", "content": "hi"}])
        body = srv.requests[-1]["payload"]
        assert "tools" not in body, "Часть 1: нативный параметр tools= запрещён"
        assert "functions" not in body
