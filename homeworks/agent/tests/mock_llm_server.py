"""Детерминированный mock OpenAI-совместимого Chat Completions сервера.

Поднимается в отдельном потоке на localhost, отдаёт заранее заданные ответы
ассистента и записывает все полученные запросы. Нужен, чтобы тестировать
Часть 1 (LLMClient + tool loop) без реальной модели, ключа и сети.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # noqa: ANN002
        pass  # тишина в тестах

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length).decode("utf-8")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"_raw": raw}

        server: "MockLLMServer" = self.server  # type: ignore[assignment]
        server.requests.append(
            {
                "path": self.path,
                "headers": dict(self.headers),
                "payload": payload,
            }
        )

        # Следующий ответ из скрипта (последний повторяется, если скрипт кончился).
        idx = min(server.step, len(server.script) - 1)
        content = server.script[idx] if server.script else ""
        server.step += 1

        body = json.dumps(
            {"choices": [{"message": {"role": "assistant", "content": content}}]}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class MockLLMServer:
    """Контекст-менеджер: with MockLLMServer(script) as srv: ... srv.base_url."""

    def __init__(self, script: list[str] | None = None):
        self.script = list(script or [])
        self.step = 0
        self.requests: list[dict] = []
        self._httpd = HTTPServer(("127.0.0.1", 0), _Handler)
        self._httpd.script = self.script  # type: ignore[attr-defined]
        self._httpd.step = 0  # type: ignore[attr-defined]
        self._httpd.requests = self.requests  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def port(self) -> int:
        return self._httpd.server_address[1]

    @property
    def base_url(self) -> str:
        # base_url для OpenAI-совместимого API оканчивается на /v1.
        return f"http://127.0.0.1:{self.port}/v1"

    # Прокидываем step туда-сюда между объектом и httpd.
    def __enter__(self) -> "MockLLMServer":
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:  # noqa: ANN002
        self.step = self._httpd.step  # type: ignore[attr-defined]
        self._httpd.shutdown()
        self._httpd.server_close()

    def sync(self) -> None:
        """Подтянуть счётчик шагов из httpd (для проверок в тесте)."""
        self.step = self._httpd.step  # type: ignore[attr-defined]


class ScriptedLLM:
    """Оффлайн-замена LLMClient: отдаёт ответы из скрипта по очереди.

    Совместима по интерфейсу (`.chat(messages) -> str`), пишет историю messages.
    Удобна для тестов tool loop / агента без сети.
    """

    def __init__(self, script: list[str]):
        self.script = list(script)
        self.step = 0
        self.calls: list[list[dict]] = []

    def chat(self, messages: list[dict]) -> str:
        self.calls.append([dict(m) for m in messages])
        out = self.script[min(self.step, len(self.script) - 1)]
        self.step += 1
        return out
