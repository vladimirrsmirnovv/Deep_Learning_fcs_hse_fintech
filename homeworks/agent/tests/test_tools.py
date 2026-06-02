"""Тесты Части 2: файловые инструменты FileTools (песочница workdir)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import FileTools  # noqa: E402


@pytest.fixture
def fs(tmp_path):
    return FileTools(str(tmp_path))


def test_write_then_read_with_line_numbers(fs):
    fs.write_file("a.txt", "первая\nвторая\nтретья")
    out = fs.read_file("a.txt")
    assert out.splitlines() == ["1\tпервая", "2\tвторая", "3\tтретья"]


def test_read_offset_limit(fs):
    fs.write_file("a.txt", "l1\nl2\nl3\nl4\nl5")
    out = fs.read_file("a.txt", offset=1, limit=2)
    assert out.splitlines() == ["2\tl2", "3\tl3"]


def test_read_missing_file(fs):
    assert "не найден" in fs.read_file("nope.txt")


def test_write_creates_subdirs(fs):
    msg = fs.write_file("pkg/mod.py", "x = 1\n")
    assert msg.startswith("OK")
    assert "pkg/mod.py" in fs.list_files()


def test_edit_exact_unique_match(fs):
    fs.write_file("c.py", "DEBUG = True\nKEY = 'secret'\n")
    msg = fs.edit_file("c.py", "DEBUG = True", "DEBUG = False")
    assert msg.startswith("OK")
    body = fs.read_file("c.py")
    assert "DEBUG = False" in body
    assert "KEY = 'secret'" in body  # остальное не затёрто


def test_edit_rejects_missing_old(fs):
    fs.write_file("c.py", "a = 1\n")
    msg = fs.edit_file("c.py", "b = 2", "b = 3")
    assert "не найдена" in msg
    assert fs.read_file("c.py").endswith("a = 1")  # файл не тронут


def test_edit_rejects_non_unique_old(fs):
    fs.write_file("c.py", "x = 1\nx = 1\n")
    msg = fs.edit_file("c.py", "x = 1", "x = 2")
    assert "не уникальна" in msg


def test_list_files_sorted_and_skips_pycache(fs):
    fs.write_file("b.py", "")
    fs.write_file("a.py", "")
    fs.write_file("__pycache__/junk.pyc", "")
    listing = fs.list_files().splitlines()
    assert listing == ["a.py", "b.py"]


def test_sandbox_escape_blocked_read(fs):
    assert "Ошибка" in fs.read_file("../../../etc/passwd")


def test_sandbox_escape_blocked_write(fs):
    msg = fs.write_file("../escape.txt", "pwn")
    assert "Ошибка" in msg


def test_run_tests_reports_pass(fs):
    fs.write_file("test_ok.py", "def test_ok():\n    assert 1 + 1 == 2\n")
    out = fs.run_tests()
    assert out.startswith("PASSED")


def test_run_tests_reports_fail_with_output(fs):
    fs.write_file("test_bad.py", "def test_bad():\n    assert 1 + 1 == 3\n")
    out = fs.run_tests()
    assert out.startswith("FAILED")
    assert "test_bad" in out  # хвост вывода pytest присутствует
