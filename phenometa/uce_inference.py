#!/usr/bin/env python3
"""Small adapter around the official UCE evaluator.

This avoids UCE's argparse `type=bool` behavior and defaults to no cell filtering,
so split membership and cell identity are preserved for leakage-safe evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import gc
import os
import sys
import tarfile
from pathlib import Path

import requests
import anndata as ad
import torch


UCE_ASSETS = {
    # filename: (official Figshare API URL, exact bytes, official MD5)
    "species_chrom.csv": ("https://api.figshare.com/v2/file/download/42706558", 4097781, "42a9b5871b1ed955fd05414b3ab89081"),
    "species_offsets.pkl": ("https://api.figshare.com/v2/file/download/42706555", 139, "ddf9143508857e62607f323b338ff383"),
    "all_tokens.torch": ("https://api.figshare.com/v2/file/download/42706585", 2979205876, "3523ea431c403f95701b6eeb0c30b997"),
    "4layer_model.torch": ("https://api.figshare.com/v2/file/download/42706576", 3403514339, "3e2f59d6da6eaa5396aa297edfcec08f"),
    "protein_embeddings.tar.gz": ("https://api.figshare.com/v2/file/download/42715213", 2735410523, "6756d7fa8348e0f70b5e5b11e324f023"),
}


def file_md5(path: Path):
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_file(url: str, destination: Path, expected: int, expected_md5: str):
    """Download atomically with Range resume and official size/MD5 checks."""
    if (destination.is_file() and destination.stat().st_size == expected
            and file_md5(destination) == expected_md5):
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    offset = temporary.stat().st_size if temporary.exists() else 0
    if offset > expected:
        temporary.unlink()
        offset = 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    print(f"Downloading UCE asset: {destination.name} ({offset}/{expected} bytes)", flush=True)
    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            offset = 0
        mode = "ab" if offset else "wb"
        written = offset
        with temporary.open(mode) as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    written += len(chunk)
    if written != expected:
        raise IOError(f"incomplete UCE download {destination.name}: {written}/{expected} bytes")
    if file_md5(temporary) != expected_md5:
        raise IOError(f"MD5 mismatch for UCE asset: {destination.name}")
    os.replace(temporary, destination)


def ensure_uce_assets(source: Path):
    model_files = source / "model_files"
    ready = model_files / ".uce_assets_v5.ready"
    if ready.is_file():
        return
    for filename, (url, size, md5) in UCE_ASSETS.items():
        download_file(url, model_files / filename, size, md5)
    protein_dir = model_files / "protein_embeddings"
    required = protein_dir / "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt"
    if not required.is_file() or required.stat().st_size == 0:
        archive = model_files / "protein_embeddings.tar.gz"
        print(f"Extracting UCE asset: {archive.name}", flush=True)
        with tarfile.open(archive, "r:gz") as handle:
            handle.extractall(model_files)
    if not required.is_file() or required.stat().st_size == 0:
        raise FileNotFoundError(f"UCE human protein embeddings missing after extraction: {required}")
    ready.write_text("Figshare article 24320806 v5 assets verified\n")


SPECIES_EMBEDDING_FILES = {
    "human": "Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt",
    "mouse": "Mus_musculus.GRCm39.gene_symbol_to_embedding_ESM2.pt",
    "frog": "Xenopus_tropicalis.Xenopus_tropicalis_v9.1.gene_symbol_to_embedding_ESM2.pt",
    "zebrafish": "Danio_rerio.GRCz11.gene_symbol_to_embedding_ESM2.pt",
    "mouse_lemur": "Microcebus_murinus.Mmur_3.0.gene_symbol_to_embedding_ESM2.pt",
    "pig": "Sus_scrofa.Sscrofa11.1.gene_symbol_to_embedding_ESM2.pt",
    "macaca_fascicularis": "Macaca_fascicularis.Macaca_fascicularis_6.0.gene_symbol_to_embedding_ESM2.pt",
    "macaca_mulatta": "Macaca_mulatta.Mmul_10.gene_symbol_to_embedding_ESM2.pt",
}


def _embedding_gene_symbols(model_files: Path, species: str):
    filename = SPECIES_EMBEDDING_FILES.get(species)
    if filename is None:
        raise ValueError(
            f"Automatic UCE gene-column selection is not configured for species={species!r}; "
            "pass --uce-gene-col explicitly"
        )
    path = model_files / "protein_embeddings" / filename
    payload = torch.load(path, map_location="cpu")
    symbols = {str(key).strip().lower() for key in payload}
    del payload
    gc.collect()
    return symbols


def _candidate_gene_names(adata, requested: str):
    if requested != "auto":
        if requested in {"index", "var_names"}:
            return [("var_names", adata.var_names.astype(str))]
        if requested not in adata.var:
            raise KeyError(f"UCE gene column {requested!r} not found; available={list(adata.var.columns)}")
        return [(requested, adata.var[requested].astype(str))]

    preferred = ["GeneSymbol", "gene_symbol", "symbol", "feature_name", "gene_name", "gene", "var_names"]
    candidates = [("var_names", adata.var_names.astype(str))]
    seen = {"var_names"}
    for name in preferred:
        if name in adata.var and name not in seen:
            candidates.append((name, adata.var[name].astype(str)))
            seen.add(name)
    return candidates


def prepare_uce_gene_names(input_path: Path, output_dir: Path, model_files: Path, species: str, requested: str):
    """Select gene symbols without modifying the user's source H5AD."""
    backed = ad.read_h5ad(input_path, backed="r")
    symbols = _embedding_gene_symbols(model_files, species)
    scored = []
    for name, values in _candidate_gene_names(backed, requested):
        cleaned = [str(value).strip() for value in values]
        overlap = sum(bool(value) and value.lower() in symbols for value in cleaned)
        unique = len(cleaned) == len(set(cleaned))
        scored.append((overlap, unique, name, cleaned))
    scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
    overlap, unique, selected, names = scored[0]
    overlap_report = {name: count for count, _unique, name, _names in scored}
    print(f"UCE gene-column selection: selected={selected} overlap={overlap}/{backed.n_vars} candidates={overlap_report}", flush=True)
    if overlap == 0:
        raise ValueError(f"No input genes match UCE {species} embeddings; candidate overlaps={overlap_report}")
    if not unique:
        raise ValueError(f"UCE gene column {selected!r} contains duplicate names; choose a unique symbol column")
    if selected == "var_names":
        return input_path

    prepared_dir = output_dir / "prepared_input"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    prepared_path = prepared_dir / input_path.name
    adata = ad.read_h5ad(input_path)
    adata.var["uce_original_var_name"] = adata.var_names.astype(str)
    adata.var_names = names
    adata.uns["uce_gene_column"] = selected
    adata.uns["uce_gene_overlap"] = int(overlap)
    temporary = prepared_path.with_suffix(".tmp.h5ad")
    adata.write_h5ad(temporary)
    os.replace(temporary, prepared_path)
    del adata
    gc.collect()
    return prepared_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--uce-source-dir", required=True)
    parser.add_argument("--adata-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--species", default="human")
    parser.add_argument("--gene-column", default="auto")
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--nlayers", type=int, default=4)
    parser.add_argument("--sample-size", type=int, default=1024)
    parser.add_argument("--model-loc")
    parser.add_argument("--filter", action="store_true")
    cli = parser.parse_args()

    source = Path(cli.uce_source_dir).resolve()
    output = Path(cli.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(source)
    sys.path.insert(0, str(source))
    ensure_uce_assets(source)
    from accelerate import Accelerator
    from evaluate import AnndataProcessor

    model_files = source / "model_files"
    original_adata_path = Path(cli.adata_path).resolve()
    prepared_adata_path = prepare_uce_gene_names(
        original_adata_path, output, model_files, cli.species, cli.gene_column
    )
    args = argparse.Namespace(
        adata_path=str(prepared_adata_path),
        dir=str(output) + os.sep,
        species=cli.species,
        filter=cli.filter,
        skip=False,
        model_loc=cli.model_loc,
        batch_size=cli.batch_size,
        pad_length=1536,
        pad_token_idx=0,
        chrom_token_left_idx=1,
        chrom_token_right_idx=2,
        cls_token_idx=3,
        CHROM_TOKEN_OFFSET=143574,
        sample_size=cli.sample_size,
        CXG=True,
        nlayers=cli.nlayers,
        output_dim=1280,
        d_hid=5120,
        token_dim=5120,
        multi_gpu=False,
        spec_chrom_csv_path=str(model_files / "species_chrom.csv"),
        token_file=str(model_files / "all_tokens.torch"),
        protein_embeddings_dir=str(model_files / "protein_embeddings") + os.sep,
        offset_pkl_path=str(model_files / "species_offsets.pkl"),
    )
    accelerator = Accelerator(project_dir=args.dir)
    processor = AnndataProcessor(args, accelerator)
    processor.preprocess_anndata()
    processor.generate_idxs()
    processor.run_evaluation()


if __name__ == "__main__":
    main()
