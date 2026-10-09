"""
extract_embeddings.py
---------------------
Zero-shot extraction of Geneformer embeddings from a single-cell AnnData (.h5ad) file.

Extracts two sets of embeddings from the final transformer layer (emb_layer=0):
  - cell_embeddings.csv  : one row per cell, shape (N_cells, 1152)
                           Each row is the mean-pool of all gene embeddings in that cell.
  - gene_embeddings.csv  : one row per gene (mean-pooled across cells), shape (N_genes, 1152).
                           With --all_layers: one file per layer named gene_embeddings_layer{N}.csv,
                           where layer 0 is the pre-transformer embedding and layers 1–N are
                           transformer layer outputs.
                           With --cell_type: gene embeddings are computed only from cells of that
                           type; cell embeddings still use all cells.
                           With --save_path PATH: written instead in scGPT's format, PATH.npz
                           (key 'layer_gene_embs', (n_layers, n_genes, 1152) float32, and
                           'n_cells', the cells each gene is averaged over) and
                           PATH_gene_names.txt.
  - cell_embeddings.csv  : with --all_layers_cell: one file per layer named
                           cell_embeddings_layer{N}.csv.

A gene's embedding at a layer is the mean of its token's hidden state over the cells
whose input contains it (Geneformer's input holds only the expressed genes, rank-ordered,
up to 4096 tokens). Sums are kept per gene and layer on the GPU, so memory does not grow
with the number of cells. <cls> and <eos> are not genes and are left out.

Prerequisites
-------------
- AnnData must have:
    adata.var["ensembl_id"]  : Ensembl IDs for each gene (e.g. "ENSG00000139618")
    adata.obs["n_counts"]    : total raw UMI counts per cell
- Raw (un-normalised) counts are required; do NOT pass log-normalised data.
- The Geneformer package must be installed: cd ~/geneformer/Geneformer && pip install .

Usage
-----
python extract_embeddings.py \\
    --input path/to/data.h5ad \\
    --output_dir ./embeddings \\
    --model_dir ~/geneformer/Geneformer \\
    [--batch_size 10] \\
    [--max_ncells 500] \\
    [--nproc 4] \\
    [--no_gene_embs] \\
    [--no_cell_embs] \\
    [--all_layers] \\
    [--all_layers_cell] \\
    [--cell_type "L2/3 IT"] \\
    [--cell_type_col cell_type] \\
    [--var_index gene_name] \\
    [--save_path path/to/embeddings/T_cell]

Arguments
---------
--input          Path to the .h5ad file.
--output_dir     Directory where output CSVs will be written (created if needed).
--model_dir      Root of the cloned Geneformer repo (contains model.safetensors and
                 the geneformer/ package). Defaults to ./Geneformer.
--batch_size     Forward-pass batch size (cells per batch). Reduce if OOM. Default: 10.
--max_ncells     Max cells to process. Pass 0 to use all cells. Default: 0 (all).
--nproc          CPU processes for dataset operations. Default: 4.
--no_gene_embs     Skip gene embedding extraction (only extract cell embeddings).
--no_cell_embs     Skip cell embedding extraction (only extract gene embeddings).
--all_layers       Extract gene embeddings from every layer; writes gene_embeddings_layer{N}.csv.
--all_layers_cell  Extract cell embeddings from every layer; writes cell_embeddings_layer{N}.csv.
--cell_type        If set, gene embeddings are extracted only from cells matching this label.
                   Cell embeddings still use all cells.
--cell_type_col    adata.obs column containing cell type labels (default: cell_type).
--var_index        adata.var column to use as the gene index in output CSVs.
                   Use 'var_names' to use adata.var_names as the index.
                   Default: Ensembl IDs from the Geneformer token dictionary.
--save_path        Write gene embeddings as PATH.npz + PATH_gene_names.txt (scGPT's format)
                   instead of CSVs.
--save_npz         With --all_layers_cell: also stack the cell embeddings into layer_cell_embs.npz.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import anndata as ad


def validate_adata(adata_path: Path, cell_type: str = None, cell_type_col: str = "cell_type", var_index: str = None) -> None:
    """Check that the AnnData has the required columns before tokenization."""
    print(f"Loading {adata_path} for validation...")
    adata = ad.read_h5ad(adata_path)
    errors = []
    if "ensembl_id" not in adata.var.columns:
        errors.append(
            "  - adata.var is missing 'ensembl_id' column.\n"
            "    Add it with: adata.var['ensembl_id'] = adata.var_names  (if var_names are Ensembl IDs)\n"
            "    or convert gene symbols → Ensembl IDs via Ensembl BioMart."
        )
    if "n_counts" not in adata.obs.columns:
        errors.append(
            "  - adata.obs is missing 'n_counts' column.\n"
            "    Add it with: adata.obs['n_counts'] = adata.X.sum(axis=1)  (raw counts required)"
        )
    if cell_type is not None:
        if cell_type_col not in adata.obs.columns:
            errors.append(
                f"  - adata.obs is missing '{cell_type_col}' column (required for --cell_type filter).\n"
                f"    Available columns: {list(adata.obs.columns)}"
            )
        elif cell_type not in adata.obs[cell_type_col].values:
            available = sorted(adata.obs[cell_type_col].unique())
            errors.append(
                f"  - Cell type '{cell_type}' not found in adata.obs['{cell_type_col}'].\n"
                f"    Available values: {available}"
            )
        else:
            n_matching = (adata.obs[cell_type_col] == cell_type).sum()
            print(f"  Cell type filter: '{cell_type}' matches {n_matching} / {len(adata)} cells")
    if var_index is not None and var_index != "var_names":
        if var_index not in adata.var.columns:
            errors.append(
                f"  - adata.var is missing '{var_index}' column (required for --var_index).\n"
                f"    Available columns: {list(adata.var.columns)}\n"
                f"    Use --var_index var_names to use adata.var_names as the gene index."
            )
    if errors:
        print("ERROR: Input AnnData is missing required fields:\n")
        for e in errors:
            print(e)
        sys.exit(1)
    n_cells, n_genes = adata.shape
    print(f"  Validated: {n_cells} cells x {n_genes} genes")
    del adata


def token_labels(adata_path: Path, var_index: str = None):
    """Geneformer token id -> output label for the input's genes, in adata.var order.

    Ensembl IDs are mapped the way the tokenizer maps them (upper-cased, through the
    Ensembl mapping dictionary; genes outside the token dictionary are dropped). The label
    is the var_index column, adata.var_names for 'var_names', or the mapped Ensembl ID by
    default. If several genes collapse into one token, it is named after the gene whose own
    ID it is, else the first one.

    Returns (tokens, labels): token ids in var order and {token id: label}.
    """
    import pickle

    from geneformer import ENSEMBL_MAPPING_FILE, TOKEN_DICTIONARY_FILE

    with open(ENSEMBL_MAPPING_FILE, "rb") as f:
        mapping = pickle.load(f)
    with open(TOKEN_DICTIONARY_FILE, "rb") as f:
        token_dict = pickle.load(f)
    var = ad.read_h5ad(adata_path, backed="r").var
    ensembl = var["ensembl_id"].astype(str).str.upper()
    if var_index is None:
        names = [None] * len(var)
    elif var_index == "var_names":
        names = var.index.astype(str)
    else:
        names = var[var_index].astype(str)

    tokens, labels = [], {}
    for ens, name in zip(ensembl, names):
        mapped = mapping.get(ens)
        token = token_dict.get(mapped) if mapped is not None else None
        if token is None:
            continue
        if token not in labels:
            tokens.append(token)
            labels[token] = mapped if name is None else name
        elif ens == mapped and name is not None:
            labels[token] = name
    return tokens, labels


def gene_embeddings(model_dir: Path, dataset_path: Path, batch_size: int, all_layers: bool,
                    max_ncells: int = None):
    """Mean hidden state of each gene token over the cells whose input contains it.

    Runs the model's own forward pass; per layer, the hidden states of the gene tokens
    (not padding, <cls> or <eos>) are added into a (vocab, d) sum on the model's device,
    so memory does not grow with the number of cells. hidden_states[0] is the input
    embedding (gene token + rank position), 1..N the transformer layers; only the last
    one unless all_layers. max_ncells subsamples as EmbExtractor does (shuffle, seed 42).

    Returns (embs, tokens, n_cells, n_total): embs (n_layers, n_tokens, d) float32 for the
    tokens seen in at least one cell, their token ids, the number of cells each was
    averaged over, and the number of cells run.
    """
    import pickle

    import torch
    from datasets import load_from_disk
    from tqdm.auto import trange

    from geneformer import TOKEN_DICTIONARY_FILE
    from geneformer import perturber_utils as pu

    with open(TOKEN_DICTIONARY_FILE, "rb") as f:
        token_dict = pickle.load(f)
    model = pu.load_model("Pretrained", 0, str(model_dir), mode="eval")
    device = next(model.parameters()).device
    pad_id = token_dict["<pad>"]
    special = torch.tensor([v for k, v in token_dict.items() if k.startswith("<")], device=device)

    data = load_from_disk(str(dataset_path))
    if max_ncells is not None and len(data) > max_ncells:
        data = data.shuffle(seed=42).select(range(max_ncells))
    data = data.sort("length", reverse=True)        # similar lengths per batch: less padding

    n_layers = model.config.num_hidden_layers + 1
    layers = list(range(n_layers)) if all_layers else [n_layers - 1]
    sums = torch.zeros(len(layers), model.config.vocab_size, model.config.hidden_size,
                       dtype=torch.float64, device=device)
    counts = torch.zeros(model.config.vocab_size, dtype=torch.float64, device=device)

    for start in trange(0, len(data), batch_size, desc="  Gene embeddings"):
        seqs = data.select(range(start, min(start + batch_size, len(data))))["input_ids"]
        lengths = torch.tensor([len(s) for s in seqs], device=device)
        ids = torch.full((len(seqs), int(lengths.max())), pad_id, dtype=torch.long, device=device)
        for row, seq in enumerate(seqs):
            ids[row, :len(seq)] = torch.tensor(seq, device=device)
        attention = torch.arange(ids.shape[1], device=device) < lengths[:, None]
        with torch.no_grad():
            hidden = model(input_ids=ids, attention_mask=attention.long()).hidden_states
        if len(hidden) != n_layers:
            raise RuntimeError(f"model returned {len(hidden)} hidden states, expected {n_layers}")
        genes = attention & ~torch.isin(ids, special)
        flat = ids[genes]
        counts.index_add_(0, flat, torch.ones(len(flat), dtype=counts.dtype, device=device))
        for out, layer in enumerate(layers):
            sums[out].index_add_(0, flat, hidden[layer][genes].to(sums.dtype))
        del hidden

    seen = torch.nonzero(counts).squeeze(1)
    embs = (sums[:, seen] / counts[seen, None]).float().cpu().numpy()
    return embs, seen.cpu().numpy(), counts[seen].long().cpu().numpy(), len(data)


def main():
    parser = argparse.ArgumentParser(
        description="Zero-shot Geneformer embedding extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--input", required=True, type=Path, help="Path to .h5ad file")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("./embeddings"),
        help="Output directory for CSV files (default: ./embeddings)",
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path("./Geneformer"),
        help="Path to cloned Geneformer repo (default: ./Geneformer)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Forward-pass batch size in cells (default: 10; reduce if OOM)",
    )
    parser.add_argument(
        "--max_ncells",
        type=int,
        default=0,
        help="Max cells to embed (default: 0 = all cells)",
    )
    parser.add_argument(
        "--nproc",
        type=int,
        default=4,
        help="CPU processes for HuggingFace dataset ops (default: 4)",
    )
    parser.add_argument(
        "--no_gene_embs",
        action="store_true",
        help="Skip gene embedding extraction (saves time/disk for large datasets)",
    )
    parser.add_argument(
        "--no_cell_embs",
        action="store_true",
        help="Skip cell embedding extraction (with --cell_type, the full file is then not tokenized)",
    )
    parser.add_argument(
        "--all_layers",
        action="store_true",
        help=(
            "Extract gene embeddings from every layer (layer 0 = pre-transformer embedding, "
            "layers 1–N = transformer outputs). Writes one CSV per layer: "
            "gene_embeddings_layer{N}.csv. Cannot be combined with --no_gene_embs."
        ),
    )
    parser.add_argument(
        "--all_layers_cell",
        action="store_true",
        help=(
            "Extract cell embeddings from every layer (layer 0 = pre-transformer embedding, "
            "layers 1–N = transformer outputs). Writes one CSV per layer: "
            "cell_embeddings_layer{N}.csv."
        ),
    )
    parser.add_argument(
        "--cell_type",
        type=str,
        default=None,
        help=(
            "If set, gene embeddings are extracted only from cells whose cell type label "
            "matches this value. Cell embeddings still use all cells. "
            "Example: --cell_type 'L2/3 IT'"
        ),
    )
    parser.add_argument(
        "--cell_type_col",
        type=str,
        default="cell_type",
        help="adata.obs column containing cell type labels (default: cell_type)",
    )
    parser.add_argument(
        "--var_index",
        type=str,
        default=None,
        help=(
            "adata.var column to use as the gene index in output CSVs. "
            "Use 'var_names' to use adata.var_names. "
            "Default: Ensembl IDs from the Geneformer token dictionary."
        ),
    )
    parser.add_argument(
        "--save_npz",
        action="store_true",
        help=(
            "With --all_layers_cell: also stack the cell embeddings into layer_cell_embs.npz "
            "(key 'layer_cell_embs', (n_layers+1, n_cells, d_model); cell ids in 'cell_ids'). "
            "For gene embeddings in .npz use --save_path."
        ),
    )
    parser.add_argument(
        "--save_path",
        type=Path,
        default=None,
        help=(
            "Write gene embeddings in scGPT's format instead of CSVs: PATH.npz with key "
            "'layer_gene_embs', (n_layers, n_genes, d_model) float32 (all n_layers+1 with "
            "--all_layers, else only the final layer), and 'n_cells' (cells averaged per gene); "
            "gene names, one per line, in PATH_gene_names.txt."
        ),
    )
    parser.add_argument(
        "--allow_cpu",
        action="store_true",
        help="Allow CPU inference (patches CUDA calls). For testing only — very slow.",
    )
    args = parser.parse_args()

    # --- CPU patching (must happen before any torch/geneformer imports) ---
    if args.allow_cpu:
        import torch
        if not torch.cuda.is_available():
            print("WARNING: --allow_cpu set; running on CPU (slow, for testing only)")
            # EmbExtractor hardcodes device="cuda" in many places; intercept all of them.
            torch.nn.Module.cuda = lambda self, *a, **kw: self
            torch.Tensor.cuda    = lambda self, *a, **kw: self

            _orig_tensor = torch.tensor
            def _safe_tensor(data, **kwargs):
                if "device" in kwargs and str(kwargs["device"]).startswith("cuda"):
                    kwargs["device"] = "cpu"
                return _orig_tensor(data, **kwargs)
            torch.tensor = _safe_tensor

            _orig_mod_to = torch.nn.Module.to
            def _safe_mod_to(self, *args, **kwargs):
                args = tuple(
                    "cpu" if isinstance(a, (str, torch.device)) and str(a).startswith("cuda") else a
                    for a in args
                )
                if "device" in kwargs and str(kwargs["device"]).startswith("cuda"):
                    kwargs["device"] = "cpu"
                return _orig_mod_to(self, *args, **kwargs)
            torch.nn.Module.to = _safe_mod_to

            _orig_t_to = torch.Tensor.to
            def _safe_t_to(self, *args, **kwargs):
                args = tuple(
                    "cpu" if isinstance(a, (str, torch.device)) and str(a).startswith("cuda") else a
                    for a in args
                )
                if "device" in kwargs and str(kwargs["device"]).startswith("cuda"):
                    kwargs["device"] = "cpu"
                return _orig_t_to(self, *args, **kwargs)
            torch.Tensor.to = _safe_t_to

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    max_ncells = args.max_ncells if args.max_ncells > 0 else None

    # --- Sanity checks ---
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    if not model_dir.exists():
        print(f"ERROR: model_dir not found: {model_dir}")
        sys.exit(1)
    if not (model_dir / "model.safetensors").exists() and not (model_dir / "Geneformer-V2-316M").exists():
        print(f"WARNING: No model.safetensors found in {model_dir}. Make sure this is the Geneformer repo root.")
    if args.no_cell_embs and args.no_gene_embs:
        print("ERROR: --no_cell_embs and --no_gene_embs together leave nothing to extract.")
        sys.exit(1)
    if args.save_path is not None and args.no_gene_embs:
        print("ERROR: --save_path is for gene embeddings; it cannot be combined with --no_gene_embs.")
        sys.exit(1)

    validate_adata(input_path, cell_type=args.cell_type, cell_type_col=args.cell_type_col, var_index=args.var_index)

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Import Geneformer (after validation, to give clear errors if not installed) ---
    try:
        from geneformer import EmbExtractor, TranscriptomeTokenizer
    except ImportError:
        print(
            "ERROR: geneformer package not found.\n"
            f"Install it with: pip install {model_dir}"
        )
        sys.exit(1)

    # Token id → output gene label, read once before entering the temp dir.
    if not args.no_gene_embs:
        gene_tokens, gene_labels = token_labels(input_path, args.var_index)

    # The full file is tokenized for cell embeddings, and for gene embeddings without --cell_type.
    tokenize_full = not args.no_cell_embs or (not args.no_gene_embs and args.cell_type is None)

    with tempfile.TemporaryDirectory(prefix="geneformer_") as tmpdir:
        tmpdir = Path(tmpdir)
        tk = TranscriptomeTokenizer(
            custom_attr_name_dict={"__cell_id__": "__cell_id__"},
            nproc=args.nproc,
            model_version="V2",
        )

        # ------------------------------------------------------------------ #
        # Step 1: Tokenize
        # ------------------------------------------------------------------ #
        if tokenize_full:
            data_dir = tmpdir / "input_data"
            tok_dir = tmpdir / "tokenized"
            data_dir.mkdir()
            tok_dir.mkdir()

            # Write h5ad to temp dir with obs_names captured as a column so it
            # survives tokenization and can be used to align npz rows to cells.
            print("  Adding cell IDs to temp h5ad...")
            _adata_tok = ad.read_h5ad(input_path)
            _adata_tok.obs["__cell_id__"] = _adata_tok.obs_names.astype(str)
            _adata_tok.write_h5ad(data_dir / input_path.name)
            del _adata_tok

            print("\n[1/3] Tokenizing...")
            tk.tokenize_data(
                data_directory=data_dir,
                output_directory=tok_dir,
                output_prefix="dataset",
                file_format="h5ad",
            )
            dataset_path = tok_dir / "dataset.dataset"
            print(f"  Tokenized dataset saved to: {dataset_path}")
        else:
            print("\n[1/3] Skipping tokenization of the full file (--no_cell_embs with --cell_type).")

        # ------------------------------------------------------------------ #
        # Step 2: Cell embeddings (mean-pool of gene embeddings per cell)
        # ------------------------------------------------------------------ #
        if not args.no_cell_embs:
            cell_layer_desc = "all layers" if args.all_layers_cell else "final layer"
            print(f"\n[2/3] Extracting cell embeddings (mean-pool, {cell_layer_desc})...")
            cell_embex = EmbExtractor(
                model_type="Pretrained",
                emb_mode="cell",
                emb_label=["__cell_id__"],
                emb_layer="all" if args.all_layers_cell else 0,
                forward_batch_size=args.batch_size,
                max_ncells=max_ncells,
                nproc=args.nproc,
                summary_stat=None,
            )
            cell_embs = cell_embex.extract_embs(
                model_directory=str(model_dir),
                input_data_file=str(dataset_path),
                output_directory=str(output_dir),
                output_prefix="cell_embeddings",
            )
            if args.all_layers_cell:
                print(f"  Done: {len(cell_embs)} layers → {output_dir}/cell_embeddings_layer*.csv")
                if args.save_npz:
                    layers = sorted(cell_embs.keys())
                    emb_cols = cell_embs[layers[0]].select_dtypes(include="number").columns
                    arr = np.stack([cell_embs[l][emb_cols].values for l in layers], axis=0)
                    cell_ids = cell_embs[layers[0]]["__cell_id__"].tolist()
                    npz_path = output_dir / "layer_cell_embs.npz"
                    np.savez(npz_path, layer_cell_embs=arr, cell_ids=np.array(cell_ids))
                    print(f"  Saved .npz → {npz_path}  shape {arr.shape}")
                    print(f"  Cell IDs saved as 'cell_ids' key in npz")
            else:
                n_cells_out = len(cell_embs)
                n_dims = len([c for c in cell_embs.columns if str(c).isdigit()])
                print(f"  Done: {n_cells_out} cells x {n_dims} dims → {output_dir}/cell_embeddings.csv")
        else:
            print("\n[2/3] Skipping cell embeddings (--no_cell_embs set).")

        # ------------------------------------------------------------------ #
        # Step 3: Gene embeddings (contextual token embeddings, mean over cells)
        # ------------------------------------------------------------------ #
        if not args.no_gene_embs:
            # If a cell type filter is requested, subset the data and re-tokenize.
            min_cells = 1
            if args.cell_type is not None:
                print(f"\n  Filtering to cell type '{args.cell_type}' for gene embeddings...")
                adata_full = ad.read_h5ad(input_path)
                mask = adata_full.obs[args.cell_type_col] == args.cell_type
                adata_sub = adata_full[mask].copy()
                del adata_full
                n_ct_cells = len(adata_sub)
                print(f"  Subset: {n_ct_cells} cells")

                min_cells = min(10, n_ct_cells)
                gene_counts = np.asarray((adata_sub.X > 0).sum(axis=0)).flatten()
                gene_mask = gene_counts >= min_cells
                n_before = adata_sub.n_vars
                adata_sub = adata_sub[:, gene_mask].copy()
                print(f"  Gene filter (expressed in ≥{min_cells} cells): {n_before} → {adata_sub.n_vars} genes retained")

                sub_data_dir = tmpdir / "input_data_sub"
                sub_tok_dir = tmpdir / "tokenized_sub"
                sub_data_dir.mkdir()
                sub_tok_dir.mkdir()
                adata_sub.obs["__cell_id__"] = adata_sub.obs_names.astype(str)
                adata_sub.write_h5ad(sub_data_dir / input_path.name)
                del adata_sub

                tk.tokenize_data(
                    data_directory=sub_data_dir,
                    output_directory=sub_tok_dir,
                    output_prefix="dataset_sub",
                    file_format="h5ad",
                )
                gene_dataset_path = sub_tok_dir / "dataset_sub.dataset"
                print(f"  Filtered dataset tokenized: {gene_dataset_path}")
            else:
                gene_dataset_path = dataset_path

            layer_desc = "all layers" if args.all_layers else "final layer"
            ct_desc = f", cell type='{args.cell_type}'" if args.cell_type else ""
            print(f"\n[3/3] Extracting gene embeddings (contextual, {layer_desc}{ct_desc})...")
            embs, seen, n_cells, n_run = gene_embeddings(
                model_dir, gene_dataset_path, args.batch_size, args.all_layers, max_ncells)

            # Genes in adata.var order; a gene must be in the input of min(10, n) cells, which
            # the expression filter above already ensures unless the 4096-token cut dropped it.
            row = {int(t): i for i, t in enumerate(seen)}
            if args.cell_type is not None:
                min_cells = min(10, n_run)
            kept = [t for t in gene_tokens if t in row and n_cells[row[t]] >= min_cells]
            n_short = sum(1 for t in gene_tokens if t in row and n_cells[row[t]] < min_cells)
            if n_short:
                print(f"  {n_short} genes in fewer than {min_cells} cells' inputs (4096-token cut) dropped")
            if len(row) != sum(1 for t in gene_tokens if t in row):
                raise RuntimeError("tokens in the model input that are not genes of adata.var")
            embs = embs[:, [row[t] for t in kept]]
            n_cells = n_cells[[row[t] for t in kept]]
            names = [str(gene_labels[t]) for t in kept]
            print(f"  {len(names)} genes x {embs.shape[0]} layers x {embs.shape[2]} dims "
                  f"from {n_run} cells (cells per gene: median {int(np.median(n_cells))})")

            if args.save_path is not None:
                save_path = args.save_path.expanduser().resolve()
                save_path.parent.mkdir(parents=True, exist_ok=True)
                npz_path = save_path.with_name(save_path.name + ".npz")
                np.savez(npz_path, layer_gene_embs=embs, n_cells=n_cells)
                names_path = save_path.with_name(save_path.name + "_gene_names.txt")
                names_path.write_text("\n".join(names))
                print(f"  Saved .npz → {npz_path}  layer_gene_embs {embs.shape}")
                print(f"  Gene names → {names_path}")
            elif args.all_layers:
                pad_width = len(str(embs.shape[0] - 1))
                for layer in range(embs.shape[0]):
                    pd.DataFrame(embs[layer], index=names).to_csv(
                        output_dir / f"gene_embeddings_layer{str(layer).zfill(pad_width)}.csv")
                print(f"  Done: {embs.shape[0]} layers → {output_dir}/gene_embeddings_layer*.csv")
            else:
                pd.DataFrame(embs[0], index=names).to_csv(output_dir / "gene_embeddings.csv")
                print(f"  Done: {len(names)} rows → {output_dir}/gene_embeddings.csv")
        else:
            print("\n[3/3] Skipping gene embeddings (--no_gene_embs set).")

    print(f"\nAll embeddings written to: {output_dir}/")
    if not args.no_cell_embs:
        if args.all_layers_cell:
            print("  cell_embeddings_layer{N}.csv — one file per layer (layer 0 = embedding, 1–N = transformer)")
        else:
            print("  cell_embeddings.csv — shape (N_cells, 1152)")
    if not args.no_gene_embs and args.save_path is None:
        index_desc = "Ensembl ID" if args.var_index is None else args.var_index
        if args.all_layers:
            print(f"  gene_embeddings_layer{{N}}.csv — one file per layer (layer 0 = embedding, 1–N = transformer), indexed by {index_desc}")
        else:
            print(f"  gene_embeddings.csv — shape (N_genes, 1152), indexed by {index_desc}")


if __name__ == "__main__":
    main()
