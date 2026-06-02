"""Часть 2: кодовый агент поверх своего LLMClient + файловых инструментов.

CodeAgent — это «ядро» кодового агента из семинара (openclaw): LLM в цикле с
файловыми тулами. Агент получает текст задачи (issue), читает/правит файлы в
песочнице `workdir` и сам решает, когда остановиться.

Агент переиспользует харнесс из Части 1 (`run_tool_loop`, `parse_tool_calls`,
`execute_tool_calls`) — менять модель/харнесс не нужно, меняется только набор
инструментов и system prompt.

Реализуйте всё, помеченное `raise NotImplementedError`.
"""
from __future__ import annotations

from llm_client import LLMClient, run_tool_loop
from tools import FILE_TOOL_SCHEMAS, FileTools

# Доп. инструкции агенту (дописываются к system prompt со списком тулов).
# Описания конкретной ошибки НЕТ — агент диагностирует баг сам, запуская тесты.
AGENT_INSTRUCTIONS = """Ты — кодовый агент. В рабочей директории лежит проект со
сломанным кодом и тесты, которые сейчас НЕ проходят. Тебе НЕ дают описания ошибки —
ты должен найти и устранить причину сам.

Твоя задача — исправить ИСХОДНЫЙ код так, чтобы все тесты проходили.

Правила:
- Сначала запусти run_tests и прочитай трейсбек — пойми, что и почему падает.
- Изучи нужные файлы: list_files, затем read_file. Баг может быть в любом из
  исходных файлов (проект бывает из нескольких файлов) — диагностируй по тестам и коду.
- Правь точечно через edit_file (точное совпадение old -> new). Меняй только то,
  что нужно для починки; не ломай остальное.
- НЕ редактируй файлы тестов (test_*.py): правка тестов — это жульничество и
  при проверке не засчитается (оценка идёт по эталонной копии тестов).
- После правок снова запускай run_tests и повторяй, пока все тесты не станут зелёными.
- Когда run_tests показывает, что всё прошло — ответь коротким текстом без tool_call.
"""


class CodeAgent:
    """Кодовый агент: LLMClient + файловые тулы в цикле над одной песочницей."""

    def __init__(self, client: LLMClient, workdir: str, max_steps: int = 25):
        self.client = client
        self.workdir = workdir
        self.max_steps = max_steps
        self.fs = FileTools(workdir)
        # Реестр инструментов: имя -> метод FileTools.
        self.tools = {
            "read_file": self.fs.read_file,
            "write_file": self.fs.write_file,
            "edit_file": self.fs.edit_file,
            "list_files": self.fs.list_files,
            "run_tests": self.fs.run_tests,
        }
        self.schemas = FILE_TOOL_SCHEMAS

    def run(self, task: str) -> str:
        """Запустить агента на задаче `task` и вернуть его финальный ответ.

        Args:
            task: текст задачи (обычно generic: «почини, чтобы тесты проходили»).

        Returns:
            Финальный текстовый ответ агента (после починки или лимита шагов).
            Побочный эффект — изменённые файлы в self.workdir.

        Подсказка: переиспользуйте run_tool_loop из Части 1, передав туда свои
        инструменты (self.tools), их схемы (self.schemas), лимит шагов и
        AGENT_INSTRUCTIONS как system_extra.
        """
        raise NotImplementedError
