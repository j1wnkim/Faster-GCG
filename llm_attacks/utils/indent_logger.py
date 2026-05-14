import sys
from functools import partial
from typing import Callable


class _IndentContext: ...


class IndentLogger:
    def __init__(self, indent_unit="  ", stream=None, log_fn: Callable | None = None):
        self.indent_level = 0
        self.indent_unit = indent_unit
        if log_fn is None:
            self.log_fn = partial(print, file=stream or sys.stdout)
        else:
            self.log_fn = log_fn

    @property
    def current_indent(self):
        return self.indent_unit * self.indent_level

    def increase_indent(self):
        self.indent_level += 1

    def decrease_indent(self):
        self.indent_level = max(0, self.indent_level - 1)

    def indent(self):
        return _IndentContext(self)

    def log(self, message: str):
        indent_str = self.current_indent
        self.log_fn(f"{indent_str}{message}")


class _IndentContext:
    def __init__(self, logger: IndentLogger):
        self.logger = logger

    def __enter__(self):
        self.logger.increase_indent()
        return self  # Not used but required for context manager

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.decrease_indent()
