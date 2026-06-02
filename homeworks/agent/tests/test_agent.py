"""Тесты Части 2: CodeAgent (agent loop) — оффлайн, со скриптованным LLM.

Реальный прогон агента на модели делается в мини-SWE-bench (checkhw,
run_evaluation.py) с настоящим Cloud.ru. Здесь проверяем только харнесс:
что агент читает задачу, дёргает файловые тулы и реально правит файлы.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent import CodeAgent  # noqa: E402
from mock_llm_server import ScriptedLLM  # noqa: E402


def test_agent_fixes_file_via_edit(tmp_path):
    # «сломанный» файл: функция складывает вместо вычитания
    (tmp_path / "calc.py").write_text("def sub(a, b):\n    return a + b\n")

    # скрипт «модели»: посмотреть файлы -> прочитать -> починить -> ответить
    script = [
        '<tool_call>{"name": "list_files", "arguments": {}}</tool_call>',
        '<tool_call>{"name": "read_file", "arguments": {"path": "calc.py"}}</tool_call>',
        '<tool_call>{"name": "edit_file", "arguments": {"path": "calc.py", '
        '"old": "return a + b", "new": "return a - b"}}</tool_call>',
        "Готово, баг исправлен.",
    ]
    agent = CodeAgent(ScriptedLLM(script), str(tmp_path), max_steps=10)
    final = agent.run("sub(a, b) должна вычитать, а не складывать")

    assert "исправлен" in final.lower()
    assert (tmp_path / "calc.py").read_text() == "def sub(a, b):\n    return a - b\n"


def test_agent_stops_on_plain_text(tmp_path):
    agent = CodeAgent(ScriptedLLM(["тут нечего чинить"]), str(tmp_path))
    assert agent.run("ничего") == "тут нечего чинить"


def test_agent_tools_registered():
    agent = CodeAgent(ScriptedLLM([""]), ".")
    assert set(agent.tools) == {
        "read_file", "write_file", "edit_file", "list_files", "run_tests",
    }


def test_agent_iterates_with_run_tests(tmp_path):
    # сломанный модуль + тест к нему; агент должен прогнать тесты, увидеть фейл,
    # починить и снова прогнать (итеративный loop через run_tests)
    (tmp_path / "m.py").write_text("def answer():\n    return 0\n")
    (tmp_path / "test_m.py").write_text(
        "from m import answer\n\ndef test_answer():\n    assert answer() == 42\n"
    )
    script = [
        '<tool_call>{"name": "run_tests", "arguments": {}}</tool_call>',  # видит FAILED
        '<tool_call>{"name": "edit_file", "arguments": {"path": "m.py", '
        '"old": "return 0", "new": "return 42"}}</tool_call>',
        '<tool_call>{"name": "run_tests", "arguments": {}}</tool_call>',  # видит PASSED
        "Готово, тесты зелёные.",
    ]
    agent = CodeAgent(ScriptedLLM(script), str(tmp_path), max_steps=10)
    final = agent.run("Почини проект, чтобы все тесты проходили.")
    assert "зел" in final.lower() or "готов" in final.lower()
    assert (tmp_path / "m.py").read_text() == "def answer():\n    return 42\n"
