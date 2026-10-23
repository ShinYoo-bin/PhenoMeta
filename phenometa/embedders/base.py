from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class EmbeddingResult:
    path: Path
    key: str = "X"


class FoundationEmbedder(ABC):
    """Backend contract: preserve cell obs_names and return one embedding per cell."""

    name = "base"

    def __init__(self, args: Any, config: Mapping[str, Any] | None = None):
        self.args = args
        self.config = dict(config or {})

    def patient_cache_dir(self, backend: str) -> Path:
        foundation_cache_root = getattr(self.args, "foundation_cache_root", None)
        if foundation_cache_root:
            dataset = self.args.dataset or Path(self.args.input_dir).name
            root = Path(foundation_cache_root) / dataset
        else:
            root = Path(self.args.output_dir)
        return root / "shared_patient_cache" / backend

    @abstractmethod
    def prepare_split(
        self,
        raw_path: Path,
        cache_path: Path,
        *,
        repeat: int,
        fold: int,
        split: str,
        log,
    ) -> EmbeddingResult:
        raise NotImplementedError
