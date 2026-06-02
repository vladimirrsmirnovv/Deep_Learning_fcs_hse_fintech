"""Часть 2: файловые инструменты кодового агента (всё внутри песочницы workdir).

Это «руки» агента: чтение, запись, точечное редактирование файлов и прогон тестов.
Реализованы как методы класса `FileTools`, привязанного к рабочей директории.
Агент (agent.py) строит из них реестр инструментов и JSON-схемы.

Ключевые идеи (как в настоящих кодовых агентах):
- `read_file` отдаёт файл С НОМЕРАМИ СТРОК и умеет читать диапазон (offset/limit).
- `edit_file` правит по ТОЧНОМУ совпадению строки (old -> new) и требует, чтобы
  `old` встречалась в файле ровно один раз — защита от правки не того места и от
  «галлюцинированных» правок.
- `run_tests` запускает pytest в песочнице — это «глаза» агента на то, что падает.
- Все пути проверяются: выйти за пределы `workdir` нельзя.

Реализуйте всё, помеченное `raise NotImplementedError`.
"""
from __future__ import annotations

import os
import subprocess
import sys


def _tail(text: str, limit: int = 2000) -> str:
    """Оставить последние limit символов (вывод тестов бывает длинным)."""
    return text if len(text) <= limit else "...(обрезано)...\n" + text[-limit:]


class FileTools:
    """Файловые инструменты в песочнице. Все пути — относительно workdir."""

    def __init__(self, workdir: str):
        self.workdir = os.path.realpath(workdir)

    def _safe_path(self, path: str) -> str:
        """Преобразовать относительный путь в абсолютный внутри workdir.

        Должен запрещать выход за пределы песочницы (напр. через '../').

        Raises:
            ValueError: если путь ведёт за пределы песочницы.
        """
        raise NotImplementedError

    def read_file(self, path: str, offset: int = 0, limit: int | None = None) -> str:
        """Прочитать файл и вернуть его содержимое С НОМЕРАМИ СТРОК.

        Args:
            path: путь к файлу относительно workdir.
            offset: с какой строки начать (0-based).
            limit: сколько строк вернуть (None — до конца).

        Returns:
            Строки в формате "{номер}\\t{текст}" (нумерация с 1), либо текст ошибки
            (файл не найден / выход за песочницу).
        """
        raise NotImplementedError

    def write_file(self, path: str, content: str) -> str:
        """Записать content в файл path целиком (создаёт директории при нужде).

        Перезапись затирает прежнее содержимое — для точечной правки см. edit_file.
        Returns: "OK: ..." при успехе, иначе текст ошибки.
        """
        raise NotImplementedError

    def edit_file(self, path: str, old: str, new: str) -> str:
        """Заменить ровно одно вхождение old на new (как Edit в кодовых агентах).

        Требует, чтобы old встречалась в файле дословно и ровно один раз.

        Returns:
            "OK: ..." при успехе, иначе текст ошибки (старая строка не найдена /
            не уникальна / файл отсутствует).
        """
        raise NotImplementedError

    def list_files(self, subdir: str = "") -> str:
        """Рекурсивно перечислить файлы в workdir (или в поддиректории subdir).

        Возвращает относительные пути, по одному на строку (отсортированные).
        Служебные директории (__pycache__, .git, .pytest_cache) пропускаются.
        """
        raise NotImplementedError

    def run_tests(self, target: str = "") -> str:
        """Запустить pytest в рабочей директории и вернуть (усечённый) вывод.

        Это «глаза» агента: даёт увидеть, какие тесты падают и почему (трейсбек).
        Подсказка: `subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=...,
        capture_output=True, text=True, timeout=...)`; длинный вывод обрежьте `_tail`.

        Args:
            target: опционально конкретный файл/папка с тестами относительно
                workdir; по умолчанию запускаются все тесты.

        Returns:
            Строка вида "PASSED\\n..." или "FAILED\\n...<хвост вывода pytest>".
        """
        raise NotImplementedError


# JSON-схемы файловых инструментов (то, что уходит модели в system prompt).
# Можете дополнять/менять описания, но имена и аргументы лучше сохранить.
FILE_TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Прочитать файл с номерами строк. offset/limit — диапазон строк.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Записать файл целиком (перезаписывает содержимое).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Заменить ровно одно дословное вхождение old на new.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["path", "old", "new"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "Рекурсивно перечислить файлы в рабочей директории.",
            "parameters": {
                "type": "object",
                "properties": {"subdir": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Запустить pytest и увидеть, какие тесты падают (трейсбек).",
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string"}},
            },
        },
    },
]
