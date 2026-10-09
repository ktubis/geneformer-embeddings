"""
Add obs["n_counts"], each cell's total counts (the row sums of X), to .h5ad files
that lack it: Geneformer's tokenizer requires the column.

The column is written in place with h5py in anndata's on-disk format (a float64
array, encoding 'array' 0.2.0, appended to obs' column-order), as in the cc_*
files; X and everything else are left untouched, so even large files take
seconds. Files that already have obs["n_counts"] are skipped. Without --apply,
only reports what it would write.

The value does not change what Geneformer sees: the tokenizer divides every gene of
a cell by the same n_counts and then ranks the genes within the cell, so any
per-cell total gives the same tokens. (The cc_* files' own n_counts, from before
their genes were filtered, are ~6% above their row sums; they are kept.)

    python scripts/add_n_counts.py [--apply] FILE.h5ad [...]

Run it while no job reads or writes the files. If h5py cannot lock a file on the
cluster's file system, set HDF5_USE_FILE_LOCKING=FALSE. To undo it for one file:
delete obs/n_counts and drop "n_counts" from obs' column-order attribute.
"""
import argparse
import sys

import h5py
import numpy as np

ROWS_PER_CHUNK = 20000


def row_sums(X):
    """Row sums of X (csr_matrix, csc_matrix or dense) in float64, read in chunks."""
    if isinstance(X, h5py.Dataset):
        return np.concatenate([X[a:a + ROWS_PER_CHUNK].astype(np.float64).sum(1)
                               for a in range(0, X.shape[0], ROWS_PER_CHUNK)])
    enc = X.attrs.get("encoding-type")
    n_rows, n_cols = (int(s) for s in X.attrs["shape"])
    if enc == "csr_matrix":
        indptr = X["indptr"][:].astype(np.int64)
        out = np.empty(n_rows)
        for a in range(0, n_rows, ROWS_PER_CHUNK):
            b = min(a + ROWS_PER_CHUNK, n_rows)
            data = X["data"][indptr[a]:indptr[b]].astype(np.float64)
            row = np.repeat(np.arange(b - a), np.diff(indptr[a:b + 1]))
            out[a:b] = np.bincount(row, weights=data, minlength=b - a)
        return out
    if enc == "csc_matrix":
        indptr = X["indptr"][:].astype(np.int64)
        out = np.zeros(n_rows)
        for a in range(0, n_cols, ROWS_PER_CHUNK):
            b = min(a + ROWS_PER_CHUNK, n_cols)
            lo, hi = indptr[a], indptr[b]
            out += np.bincount(X["indices"][lo:hi], weights=X["data"][lo:hi].astype(np.float64),
                               minlength=n_rows)
        return out
    raise ValueError(f"X has encoding {enc!r}")


def n_obs(obs):
    return len(obs[obs.attrs["_index"]])


def add(path, apply):
    with h5py.File(path, "r+" if apply else "r") as f:
        obs = f["obs"]
        if obs.attrs.get("encoding-type") != "dataframe":
            print(f"  {path}: obs is not an anndata dataframe group; left alone")
            return False
        if "n_counts" in obs:
            print(f"  {path}: has n_counts; skipped")
            return True
        n = row_sums(f["X"])
        if len(n) != n_obs(obs):
            raise ValueError(f"{path}: {len(n)} rows in X but {n_obs(obs)} cells in obs")
        print(f"  {path}: {len(n)} cells, n_counts median {np.median(n):.0f}, min {n.min():.0f}, "
              f"{int((n == 0).sum())} with 0{'' if apply else '  (dry run)'}")
        if not apply:
            return True
        col = obs.create_dataset("n_counts", data=n)
        col.attrs["encoding-type"] = "array"
        col.attrs["encoding-version"] = "0.2.0"
        order = [str(c) for c in np.atleast_1d(obs.attrs.get("column-order", []))]
        obs.attrs["column-order"] = np.array(order + ["n_counts"], dtype=h5py.string_dtype())
    # read it back the way the tokenizer does
    import anndata as ad
    back = ad.read_h5ad(path, backed="r").obs
    if "n_counts" not in back.columns or not np.allclose(back["n_counts"].to_numpy(), n):
        raise RuntimeError(f"{path}: n_counts did not read back through anndata")
    print(f"    written and read back through anndata ({len(back.columns)} obs columns)")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--apply", action="store_true", help="write the column (default: dry run)")
    args = ap.parse_args()
    ok = all([add(p, args.apply) for p in args.files])
    if not args.apply:
        print("dry run: nothing written; rerun with --apply")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
