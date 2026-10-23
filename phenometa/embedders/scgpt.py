from pathlib import Path

from ..scgpt_embedding import embed_split_h5ad
from .base import EmbeddingResult, FoundationEmbedder
from .registry import register_embedder


@register_embedder("scgpt")
class ScGPTEmbedder(FoundationEmbedder):
    name = "scgpt"

    def prepare_split(self, raw_path: Path, cache_path: Path, *, repeat, fold, split, log):
        option = lambda name: self.config.get(name, getattr(self.args, f"scgpt_{name}"))
        output = embed_split_h5ad(
            raw_path,
            cache_path,
            checkpoint_dir=option("checkpoint_dir"),
            source_dir=option("source_dir"),
            patient_col=self.args.patient_col,
            gene_col=option("gene_col"),
            batch_size=option("batch_size"),
            max_length=option("max_length"),
            device=self.config.get("device", self.args.resolved_device),
            use_fast_transformer=option("use_fast_transformer"),
            overwrite=self.args.overwrite_foundation_cache,
            patient_cache_dir=self.patient_cache_dir("scgpt"),
            log=log,
            repeat=repeat,
            fold=fold,
            split=split,
        )
        return EmbeddingResult(Path(output), "X")
