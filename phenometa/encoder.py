"""One-time encoder boundary.

PP training consumes persisted latent embeddings and never imports or updates the
encoder.  This module only provides path bootstrapping and a safe helper for a
caller-owned pretrained encoder, including the local scGPT checkout.
"""
from pathlib import Path
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCGPT_SOURCE = Path(os.environ.get("SCGPT_SOURCE_DIR", PROJECT_ROOT / "external" / "scGPT")).expanduser()


def add_local_scgpt_to_path(source=DEFAULT_SCGPT_SOURCE):
    source = Path(source).resolve()
    package = source / "scgpt"
    if not package.exists(): raise FileNotFoundError(f"scGPT package not found: {package}")
    if str(source) not in sys.path: sys.path.insert(0, str(source))
    import scgpt
    return scgpt


@torch.no_grad()
def encode_once(encoder, cell_features, device="cuda", batch_size=256):
    """Run a frozen caller-configured encoder once and return CPU float32 latent."""
    encoder.eval(); encoder.requires_grad_(False); device = torch.device(device)
    encoder.to(device); values = torch.as_tensor(cell_features, dtype=torch.float32)
    chunks = []
    for start in range(0, values.shape[0], batch_size):
        output = encoder(values[start:start + batch_size].to(device))
        if isinstance(output, dict):
            for key in ("cell_emb", "latent", "embedding", "output"):
                if key in output: output = output[key]; break
        if isinstance(output, (tuple, list)): output = output[0]
        chunks.append(output.detach().float().cpu())
    return np.asarray(torch.cat(chunks).numpy(), dtype=np.float32)
