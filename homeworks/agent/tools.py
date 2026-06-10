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
        if path is None:
            path = ""

        if not isinstance(path, str):
            raise ValueError("путь должен быть строкой")

        candidate = os.path.realpath(os.path.join(self.workdir, path))
        if os.path.commonpath([self.workdir, candidate]) != self.workdir:
            raise ValueError(f"путь выходит за пределы рабочей директории: {path}")

        return candidate

    def read_file(self, path: str, offset: int = 0, limit: int | None = None) -> str:
        """Прочитать файл и вернуть его содержимое С НОМЕРАМИ СТРОК."""
        try:
            safe = self._safe_path(path)

            offset = 0 if offset is None else int(offset)
            if offset < 0:
                return "Ошибка: offset должен быть неотрицательным"

            if limit is not None:
                limit = int(limit)
                if limit < 0:
                    return "Ошибка: limit должен быть неотрицательным"

            with open(safe, "r", encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()

            selected = lines[offset:] if limit is None else lines[offset: offset + limit]
            return "\n".join(
                f"{line_no}\t{line}"
                for line_no, line in enumerate(selected, start=offset + 1)
            )

        except FileNotFoundError:
            return f"Ошибка: файл не найден: {path}"
        except IsADirectoryError:
            return f"Ошибка: путь указывает на директорию: {path}"
        except ValueError as exc:
            return f"Ошибка: {exc}"
        except OSError as exc:
            return f"Ошибка: не удалось прочитать {path}: {exc}"

    def write_file(self, path: str, content: str) -> str:
        """Записать content в файл path целиком (создаёт директории при нужде)."""
        try:
            safe = self._safe_path(path)

            parent = os.path.dirname(safe)
            if parent:
                os.makedirs(parent, exist_ok=True)

            with open(safe, "w", encoding="utf-8") as f:
                f.write(content)

            rel = os.path.relpath(safe, self.workdir)
            return f"OK: wrote {rel}"

        except ValueError as exc:
            return f"Ошибка: {exc}"
        except OSError as exc:
            return f"Ошибка: не удалось записать {path}: {exc}"

    def edit_file(self, path: str, old: str, new: str) -> str:
        """Заменить ровно одно вхождение old на new."""
        try:
            safe = self._safe_path(path)

            with open(safe, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()

            n_matches = content.count(old)

            if n_matches == 0:
                return "Ошибка: старая строка не найдена"

            if n_matches > 1:
                return f"Ошибка: старая строка не уникальна ({n_matches} совпадений)"

            updated = content.replace(old, new, 1)

            with open(safe, "w", encoding="utf-8") as f:
                f.write(updated)

            rel = os.path.relpath(safe, self.workdir)
            return f"OK: edited {rel}"

        except FileNotFoundError:
            return f"Ошибка: файл не найден: {path}"
        except IsADirectoryError:
            return f"Ошибка: путь указывает на директорию: {path}"
        except ValueError as exc:
            return f"Ошибка: {exc}"
        except OSError as exc:
            return f"Ошибка: не удалось отредактировать {path}: {exc}"

    def list_files(self, subdir: str = "") -> str:
        """Рекурсивно перечислить файлы в workdir (или в поддиректории subdir)."""
        skip_dirs = {
            "__pycache__",
            ".git",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
        }

        try:
            root = self._safe_path(subdir or "")

            if not os.path.exists(root):
                return f"Ошибка: директория не найдена: {subdir}"

            if not os.path.isdir(root):
                return f"Ошибка: это не директория: {subdir}"

            files: list[str] = []

            for current_root, dirnames, filenames in os.walk(root):
                dirnames[:] = sorted(d for d in dirnames if d not in skip_dirs)

                for filename in sorted(filenames):
                    full_path = os.path.join(current_root, filename)
                    rel_path = os.path.relpath(full_path, self.workdir)
                    files.append(rel_path.replace(os.sep, "/"))

            return "\n".join(sorted(files))

        except ValueError as exc:
            return f"Ошибка: {exc}"
        except OSError as exc:
            return f"Ошибка: не удалось перечислить файлы: {exc}"

    def run_tests(self, target: str = "") -> str:
        """Запустить pytest в рабочей директории и вернуть (усечённый) вывод."""
        try:
            cmd = [sys.executable, "-m", "pytest", "-q"]

            if target:
                safe_target = self._safe_path(target)
                if not os.path.exists(safe_target):
                    return f"FAILED\nОшибка: test target не найден: {target}"
                cmd.append(os.path.relpath(safe_target, self.workdir))

            completed = subprocess.run(
                cmd,
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=30,
            )

            output = (completed.stdout or "") + (completed.stderr or "")
            status = "PASSED" if completed.returncode == 0 else "FAILED"

            return f"{status}\n{_tail(output)}"

        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""

            if isinstance(stdout, bytes):
                stdout = stdout.decode("utf-8", errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode("utf-8", errors="replace")

            output = stdout + stderr
            return f"FAILED\nОшибка: pytest timed out\n{_tail(output)}"

        except ValueError as exc:
            return f"FAILED\nОшибка: {exc}"
        except OSError as exc:
            return f"FAILED\nОшибка: не удалось запустить pytest: {exc}"


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
