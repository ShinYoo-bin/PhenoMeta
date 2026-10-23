from __future__ import annotations

import importlib

from .base import FoundationEmbedder

_REGISTRY = {}


def register_embedder(name):
    def decorator(cls):
        if not issubclass(cls, FoundationEmbedder):
            raise TypeError("foundation backend must subclass FoundationEmbedder")
        _REGISTRY[name.lower()] = cls
        return cls
    return decorator


def _load_builtins():
    from . import precomputed, scgpt, uce  # noqa: F401


def available_embedders():
    _load_builtins()
    return sorted(_REGISTRY)


def build_embedder(spec, args, config=None):
    """Build a registered backend or a custom `python.module:ClassName` backend."""
    _load_builtins()
    key = spec.lower()
    if key in _REGISTRY:
        return _REGISTRY[key](args, config)
    if ":" in spec:
        module_name, class_name = spec.rsplit(":", 1)
        cls = getattr(importlib.import_module(module_name), class_name)
        if not issubclass(cls, FoundationEmbedder):
            raise TypeError(f"{spec} must subclass FoundationEmbedder")
        return cls(args, config)
    raise KeyError(f"unknown foundation model {spec!r}; available={available_embedders()}")
