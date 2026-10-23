from pathlib import Path

import anndata as ad

from .base import EmbeddingResult, FoundationEmbedder
from .registry import register_embedder


@register_embedder("precomputed")
class PrecomputedEmbedder(FoundationEmbedder):
    name = "precomputed"

    def prepare_split(self, raw_path: Path, cache_path: Path, *, repeat, fold, split, log):
        if not self.args.embedding_input_dir:
            raise ValueError("--embedding-input-dir is required for foundation-model=precomputed")
        path = Path(self.args.embedding_input_dir) / f"repeat_{repeat}" / f"fold_{fold}_{split}.h5ad"
        if not path.exists():
            raise FileNotFoundError(f"precomputed embedding not found: {path}")
        raw = ad.read_h5ad(raw_path, backed="r")
        embedded = ad.read_h5ad(path, backed="r")
        if not raw.obs_names.is_unique or not embedded.obs_names.is_unique:
            raise ValueError("raw and precomputed embedding files require unique obs_names")
        missing = raw.obs_names.difference(embedded.obs_names)
        if len(missing):
            raise KeyError(f"{len(missing)} raw cells have no precomputed embedding")
        extra = embedded.obs_names.difference(raw.obs_names)
        if len(extra):
            raise KeyError(f"{len(extra)} precomputed cells are absent from raw input")
        key = self.config.get("embedding_key", self.args.embedding_key)
        if key != "X" and key not in embedded.obsm and key not in embedded.layers:
            raise KeyError(f"embedding key {key!r} absent from {path}")
        if log:
            log.info("[%s] precomputed=%s cells=%d key=%s", self.name, path, embedded.n_obs, key)
        return EmbeddingResult(path, key)
