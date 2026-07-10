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
  - cell_embeddings.csv  : with --all_layers_cell: one file per layer named
                           cell_embeddings_layer{N}.csv.

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
    [--all_layers] \\
    [--all_layers_cell] \\
    [--cell_type "L2/3 IT"] \\
    [--cell_type_col cell_type] \\
    [--var_index gene_name]

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
                   Gene embeddings can be large: ~(N_cells * avg_genes_per_cell) rows.
--all_layers       Extract gene embeddings from every layer; writes gene_embeddings_layer{N}.csv.
--all_layers_cell  Extract cell embeddings from every layer; writes cell_embeddings_layer{N}.csv.
--cell_type        If set, gene embeddings are extracted only from cells matching this label.
                   Cell embeddings still use all cells.
--cell_type_col    adata.obs column containing cell type labels (default: cell_type).
--var_index        adata.var column to use as the gene index in output CSVs.
                   Use 'var_names' to use adata.var_names as the index.
                   Default: Ensembl IDs from the Geneformer token dictionary.
"""

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

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
            "Save gene embeddings as a single .npz file (layer_gene_embs.npz) with key "
            "'layer_gene_embs', shape (n_layers+1, n_genes, d_model) — same format as scGPT. "
            "Requires --all_layers. Gene names are written alongside as layer_gene_embs_gene_names.txt."
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

    # Build the Ensembl ID → desired index mapping once, before entering the temp dir.
    if args.var_index is not None:
        _var = ad.read_h5ad(input_path, backed="r").var
        if args.var_index == "var_names":
            id_to_label = dict(zip(_var["ensembl_id"], _var.index))
        else:
            id_to_label = dict(zip(_var["ensembl_id"], _var[args.var_index]))
        del _var

    with tempfile.TemporaryDirectory(prefix="geneformer_") as tmpdir:
        tmpdir = Path(tmpdir)
        data_dir = tmpdir / "input_data"
        tok_dir = tmpdir / "tokenized"
        data_dir.mkdir()
        tok_dir.mkdir()

        # The tokenizer expects a directory of .h5ad files
        shutil.copy2(input_path, data_dir / input_path.name)

        # ------------------------------------------------------------------ #
        # Step 1: Tokenize
        # ------------------------------------------------------------------ #
        print("\n[1/3] Tokenizing...")
        tk = TranscriptomeTokenizer(
            custom_attr_name_dict=None,
            nproc=args.nproc,
            model_version="V2",
        )
        tk.tokenize_data(
            data_directory=data_dir,
            output_directory=tok_dir,
            output_prefix="dataset",
            file_format="h5ad",
        )
        dataset_path = tok_dir / "dataset.dataset"
        print(f"  Tokenized dataset saved to: {dataset_path}")

        # ------------------------------------------------------------------ #
        # Step 2: Cell embeddings (mean-pool of gene embeddings per cell)
        # ------------------------------------------------------------------ #
        cell_layer_desc = "all layers" if args.all_layers_cell else "final layer"
        print(f"\n[2/3] Extracting cell embeddings (mean-pool, {cell_layer_desc})...")
        cell_embex = EmbExtractor(
            model_type="Pretrained",
            emb_mode="cell",
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
                arr = np.stack([cell_embs[l].values for l in layers], axis=0)
                npz_path = output_dir / "layer_cell_embs.npz"
                np.savez(npz_path, layer_cell_embs=arr)
                print(f"  Saved .npz → {npz_path}  shape {arr.shape}")
        else:
            n_cells_out = len(cell_embs)
            n_dims = len([c for c in cell_embs.columns if str(c).isdigit()])
            print(f"  Done: {n_cells_out} cells x {n_dims} dims → {output_dir}/cell_embeddings.csv")

        # ------------------------------------------------------------------ #
        # Step 3: Gene embeddings (contextual token embeddings, per cell)
        # ------------------------------------------------------------------ #
        if not args.no_gene_embs:
            # If a cell type filter is requested, subset the data and re-tokenize.
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
            gene_embex = EmbExtractor(
                model_type="Pretrained",
                emb_mode="gene",
                gene_emb_style="mean_pool",
                emb_layer="all" if args.all_layers else 0,
                forward_batch_size=args.batch_size,
                max_ncells=max_ncells,
                nproc=args.nproc,
                summary_stat=None,
            )
            gene_embs = gene_embex.extract_embs(
                model_directory=str(model_dir),
                input_data_file=str(gene_dataset_path),
                output_directory=str(output_dir),
                output_prefix="gene_embeddings",
            )

            # Remap Ensembl ID index to the requested var column / var_names and re-save.
            if args.var_index is not None:
                if args.all_layers:
                    n_layers = len(gene_embs)
                    pad_width = len(str(n_layers - 1))
                    for layer_key in sorted(gene_embs.keys()):
                        df = gene_embs[layer_key]
                        df.index = [id_to_label.get(g, g) for g in df.index]
                        df.to_csv(output_dir / f"gene_embeddings_layer{str(layer_key).zfill(pad_width)}.csv")
                else:
                    gene_embs.index = [id_to_label.get(g, g) for g in gene_embs.index]
                    gene_embs.to_csv(output_dir / "gene_embeddings.csv")

            # Save as stacked .npz matching scGPT's layer_gene_embs format.
            if args.save_npz:
                if not args.all_layers:
                    print("  WARNING: --save_npz requires --all_layers; skipping .npz output.")
                else:
                    layers = sorted(gene_embs.keys())
                    arr = np.stack([gene_embs[l].values for l in layers], axis=0)
                    gene_names = gene_embs[layers[0]].index.tolist()
                    npz_path = output_dir / "layer_gene_embs.npz"
                    np.savez(npz_path, layer_gene_embs=arr)
                    names_path = output_dir / "layer_gene_embs_gene_names.txt"
                    names_path.write_text("\n".join(gene_names))
                    print(f"  Saved .npz → {npz_path}  shape {arr.shape}")
                    print(f"  Gene names → {names_path}")

            if args.all_layers:
                print(f"  Done: {len(gene_embs)} layers → {output_dir}/gene_embeddings_layer*.csv")
            else:
                print(f"  Done: {len(gene_embs)} rows → {output_dir}/gene_embeddings.csv")
        else:
            print("\n[3/3] Skipping gene embeddings (--no_gene_embs set).")

    print(f"\nAll embeddings written to: {output_dir}/")
    if args.all_layers_cell:
        print("  cell_embeddings_layer{N}.csv — one file per layer (layer 0 = embedding, 1–N = transformer)")
    else:
        print("  cell_embeddings.csv — shape (N_cells, 1152)")
    if not args.no_gene_embs:
        if args.all_layers:
            print("  gene_embeddings_layer{N}.csv — one file per layer (layer 0 = embedding, 1–N = transformer)")
        else:
            print("  gene_embeddings.csv — shape (N_genes, 1152), indexed by Ensembl ID")


if __name__ == "__main__":
    main()
