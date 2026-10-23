from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import anndata as ad
import numpy as np

from .base import EmbeddingResult, FoundationEmbedder
from ..patient_cache import assemble, cache_complete, populate, signature
from .registry import register_embedder


@register_embedder("uce")
class UCEEmbedder(FoundationEmbedder):
    """Run the official UCE evaluator in its own Python environment."""

    name = "uce"

    def option(self, name):
        return self.config.get(name, getattr(self.args, f"uce_{name}"))

    def prepare_split(self, raw_path: Path, cache_path: Path, *, repeat, fold, split, log):
        source = Path(self.option("source_dir")).resolve()
        python = Path(self.option("python")).resolve()
        runner = Path(__file__).resolve().parents[1] / "uce_inference.py"
        if not (source / "eval_single_anndata.py").is_file():
            raise FileNotFoundError(f"UCE source not found: {source}")
        if not python.is_file():
            raise FileNotFoundError(f"UCE Python not found: {python}")

        work_dir = cache_path.parent / f"{split}_work"
        output = work_dir / f"{raw_path.stem}_uce_adata.h5ad"
        patient_cache = self.patient_cache_dir("uce")
        cache_signature = signature({"backend": "uce", "species": self.option("species"),
            "gene_col": self.option("gene_col"), "nlayers": int(self.option("nlayers")),
            "sample_size": int(self.option("sample_size")), "filter": bool(self.option("filter")),
            "model_loc": str(self.option("model_loc"))})
        if output.exists() and not self.args.overwrite_foundation_cache:
            self._validate(raw_path, output)
            if not cache_complete(raw_path, patient_cache, self.args.patient_col, cache_signature):
                populate(raw_path, output, patient_cache, self.args.patient_col, "X_uce", cache_signature)
            if log:
                log.info("[uce] using cache=%s", output)
            return EmbeddingResult(output, "X_uce")

        if (not self.args.overwrite_foundation_cache and
                cache_complete(raw_path, patient_cache, self.args.patient_col, cache_signature)):
            if log: log.info("[uce patient cache] assemble split=%s", raw_path)
            assemble(raw_path, output, patient_cache, self.args.patient_col, cache_signature, output_key="X_uce")
            self._validate(raw_path, output)
            return EmbeddingResult(output, "X_uce")

        work_dir.mkdir(parents=True, exist_ok=True)
        command = [
            str(python), str(runner),
            "--uce-source-dir", str(source),
            "--adata-path", str(raw_path.resolve()),
            "--output-dir", str(work_dir.resolve()),
            "--species", str(self.option("species")),
            "--gene-column", str(self.option("gene_col")),
            "--batch-size", str(self.option("batch_size")),
            "--nlayers", str(self.option("nlayers")),
            "--sample-size", str(self.option("sample_size")),
        ]
        model_loc = self.option("model_loc")
        if model_loc:
            command.extend(["--model-loc", str(Path(model_loc).resolve())])
        if self.option("filter"):
            command.append("--filter")

        env = os.environ.copy()
        env.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "phenometa_uce_matplotlib_cache"))
        env.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "phenometa_uce_numba_cache"))
        if log:
            log.info("[uce] split=%s input=%s work_dir=%s", split, raw_path, work_dir)
        subprocess.run(command, cwd=source, env=env, check=True)
        if not output.exists():
            candidates = sorted(work_dir.glob("*_uce_adata.h5ad"))
            raise FileNotFoundError(f"UCE did not create {output}; candidates={candidates}")
        self._validate(raw_path, output)
        populate(raw_path, output, patient_cache, self.args.patient_col, "X_uce", cache_signature)
        return EmbeddingResult(output, "X_uce")

    @staticmethod
    def _validate(raw_path, output):
        raw = ad.read_h5ad(raw_path, backed="r")
        embedded = ad.read_h5ad(output, backed="r")
        if "X_uce" not in embedded.obsm:
            raise KeyError(f"X_uce absent from UCE output: {output}")
        if not raw.obs_names.is_unique or not embedded.obs_names.is_unique:
            raise ValueError("UCE input/output require unique obs_names")
        if not raw.obs_names.equals(embedded.obs_names):
            missing = raw.obs_names.difference(embedded.obs_names)
            extra = embedded.obs_names.difference(raw.obs_names)
            raise ValueError(f"UCE changed cells or order: missing={len(missing)} extra={len(extra)}")
        values = np.asarray(embedded.obsm["X_uce"])
        if values.ndim != 2 or values.shape[0] != raw.n_obs or not np.isfinite(values).all():
            raise ValueError(f"invalid UCE embedding shape/values: {values.shape}")
