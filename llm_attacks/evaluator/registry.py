"""Handle register_evaluator."""

from typing import Type

from .base import BaseEvaluator

# Example: Module-level registry
_evaluator_registry: dict[str, Type[BaseEvaluator]] = {}


def register_evaluator(name: str | None = None):
    """Decorator to register an evaluator class."""

    def decorator(cls: Type[BaseEvaluator]):
        """Register the evaluator class."""
        nonlocal name
        if name is None:
            name = cls.__name__
        if name in _evaluator_registry:
            raise ValueError(f"Evaluator {name} already registered")
        _evaluator_registry[name] = cls
        return cls

    return decorator


def get_evaluator_cls(name: str) -> Type[BaseEvaluator]:
    """Get the evaluator class by name."""
    try:
        return _evaluator_registry[name]
    except KeyError:
        raise KeyError(f"Evaluator {name} not registered")


def get_evaluator(name: str, *args, **kwargs) -> BaseEvaluator:
    """Get the evaluator instance by name."""
    try:
        return _evaluator_registry[name](*args, **kwargs)
    except KeyError:
        raise KeyError(f"Evaluator {name} not registered")
