"""
Activation addition: does Geneformer use the per-instance offset as context when
it predicts masked genes? The counterpart of scgpt_embeds'
scripts/offset_intervention.py; batches, shifts, passes and output layout are the
same, only the model, its input and its read-out differ.

An instance is one celltype of one batch. A batch is one h5ad file or a group of
them (--batch NAME=PATTERN[,PATTERN...], e.g. the five 10x 3' colon sites); the
files of a batch are pooled, so site is not a factor. Each instance is prepared
as for the gene embeddings (extract_embeddings.py --cell_type): genes expressed
in fewer than min(10, n_cells) of its cells removed, then Geneformer's V2
tokenizer (each gene divided by its corpus median, ranked within the cell;
<cls>, the top 4094 genes, <eos>). The files need var["ensembl_id"] (raw counts
in X); obs["n_counts"] is computed from X where missing (any per-cell total
gives the same tokens). There is no cell split: every cell is used both to
estimate the shifts and to evaluate them, because the shifts that are tested
come from other instances (see Shifts).

Estimation: one unmasked forward pass, as for the gene embeddings; e[g, L] =
gene g's hidden state at layer L (hidden_states[L]: 0 the embedding output,
1..18 the transformer layers) averaged over the cells whose input contains it,
for the genes in at least min(10, n_cells) inputs. A shift from instance A to B
is paired by gene: the mean over the genes both have of e_B[g, L] - e_A[g, L].

Evaluation: masked-gene prediction, the task Geneformer was pretrained on
(BertForMaskedLM, cross-entropy at masked positions; bidirectional attention).
A fixed random --heldout-ratio (default 0.15, the pretraining ratio) of each
cell's gene positions are replaced by <mask> (no 80/10/10 mix of random and
unchanged tokens); the same positions in every condition. A masked position
keeps its position embedding, i.e. its rank. Read-outs, per held-out gene g of a
cell, from the head's probabilities at the cell's masked positions:

  pred_rank   g's predicted relative rank: sum_p P(g at p) * v_p / sum_p P(g at p)
              over the masked positions p, where v_p = 1 - (rank of p) / (n - 1),
              n the cell's genes (1 = the top-ranked gene, 0 the last); the
              main read-out, comparable across cells and batches
  p_correct   P(g at its own position); falls under any context/position
              mismatch, so it shows how selectively a shift disrupts, not its
              direction
  logp_correct, top1   log P(g at its own position) (the pretraining loss) and
              whether g is the head's top gene there; quality diagnostics

Conditions: for each shift (below) and each --layers L, the shift is added at
the output of layer L (hidden_states[L]; L = 0: the embedding output) in two
passes; <cls> and <eos> are never shifted. The encoder layers are run one at a
time (run_layers; checked against the model's own forward once per run), so each
condition starts from the unshifted pass's states at its layer.

  ctx   unmasked gene tokens shifted, masked tokens not (L < 18; at L = 18 the
        head reads the masked positions only)
  dir   unmasked gene tokens and masked tokens shifted (L >= 1; at L = 0 a masked
        token is <mask> + its position, the same in every cell). At L = 18 this is
        a direct push on the head's input.

Shifts, applied to every cell of an instance, exactly as in the scGPT script;
for a batch pair A:B (--batch-pairs) and celltype k: chem_own|A>B (k's own
paired shift (A, k) -> (B, k), in-sample; the ceiling), chem_heldout|A>B (the
mean over every other celltype; the main condition), chem_heldout_x<f>|A>B
(--doses), rand_chem<i>|A>B (Gaussian, chem_heldout's norm per layer), and on
(B, k) chem_own_rev, chem_heldout_rev, chem_heldout_rev_x<f>, rand_chem<i>_rev;
for a celltype pair a:b (--celltype-pairs) in batch X: ct_own|a>b,
ct_heldout|a>b (the mean over the other batches), ct_heldout_x<f>|a>b,
rand_ct<i>|a>b. "base" is the unshifted model.

    python scripts/offset_intervention.py --data-dir /data/final_datasets \\
        --batch "3p=cc_10x3p_*colon,cc_10x3p_rectum" \\
        --batch "5p=cc_10x5p_*colon,cc_10x5p_rectum" --batch-pairs 3p:5p \\
        --celltypes "T cell" "B cell" "Epithelial cell" \\
        --celltype-pairs "T cell:B cell" --layers 0 1 9 17 18 \\
        --model-dir Geneformer-V2-316M --out offset_runs/pilot

Stages (--stage), as in the scGPT script, so that every cell can be used (no cap
by default): shifts (one job: the estimation pass over every cell; shifts.npz,
plan.pt, tasks.tsv, labels.json), eval (one job per task = an instance's chunk
of --chunk-size cells, --task or $SLURM_ARRAY_TASK_ID; parts/...), merge (CPU:
adds the parts up), all (the three in one job; the default). Every random choice
of an instance (subsample, masked positions, <cls> cells) is drawn from a seed
of its own, so every job prepares it the same way.

<out>/<batch>__<celltype>.npz per instance: genes (n,) the gene names
(adata.var_names), token_ids (n,), n_heldout (n,) how often each gene was masked
over the instance's cells, target_sum (n,) the sum of its true relative ranks
there, conditions (c,) ("base" or "L<layer>|ctx|<shift>" / "L<layer>|dir|<shift>"),
per read-out the sums over the cells per condition: pred_rank, p_correct,
logp_correct, top1 (c, n); cls_conditions (c,) and cls (c, m, d) float16, the
final <cls> output of --cls-cells random cells per condition (<cls> sees the
masked tokens, so dir passes have their own), cls_cells (m,) their indices among
the instance's cells; and <out>/shifts.npz with the same keys as the scGPT
script's: keys, n_cells, n_genes, n_genes_median (genes per cell's input),
chem_own_keys / chem_own / chem_own_n_shared, chem_heldout_keys / chem_heldout /
chem_heldout_n, ct_own_*, ct_heldout_*, layers, heldout_ratio.
"""

import argparse
import fnmatch
import json
import os
import pickle
import sys
import tempfile
import zlib
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ── Data ──────────────────────────────────────────────────────────────────────

def resolve_batches(specs, names, data_dir):
    """{batch: [h5ad paths]} for --batch NAME=PATTERN[,PATTERN...] specs (globs
    over the file names in data_dir, without .h5ad) and for every other name in
    `names`, which is a file of its own."""
    files = sorted(p.stem for p in Path(data_dir).glob("*.h5ad"))
    out = {}
    for spec in specs:
        name, sep, patterns = spec.partition("=")
        if not sep or not name or not patterns:
            raise ValueError(f"expected NAME=PATTERN[,PATTERN...], got {spec!r}")
        hits = [f for f in files if any(fnmatch.fnmatch(f, p) for p in patterns.split(","))]
        if not hits:
            raise ValueError(f"--batch {spec!r}: no h5ad file in {data_dir} matches")
        out[name] = [os.path.join(data_dir, f"{f}.h5ad") for f in hits]
    for name in names:
        if name not in out:
            path = os.path.join(data_dir, f"{name}.h5ad")
            if not os.path.exists(path):
                raise ValueError(f"{name}: neither a --batch name nor a file in {data_dir}")
            out[name] = [path]
    return out


def celltype_labels(specs):
    """{instance celltype: set of obs labels} from --celltypes entries NAME or
    NAME=LABEL1,LABEL2 (cells with any of the labels form the instance NAME, e.g.
    "PP=gamma,pp" where datasets name a celltype differently)."""
    out = {}
    for spec in specs:
        name, sep, labels = spec.partition("=")
        out[name] = set(labels.split(",")) if sep else {name}
    return out


def pool_celltypes(paths, celltypes, col):
    """{name: AnnData of its cells pooled over the files (shared genes, their
    var columns kept where the files agree)}; obs["n_counts"] = X's row sums where
    a file has none;
    for celltypes = {name: set of obs labels} (celltype_labels)."""
    parts = {ct: [] for ct in celltypes}
    for path in paths:
        adata = ad.read_h5ad(path)
        if "ensembl_id" not in adata.var.columns:
            raise ValueError(f"{path}: var has no 'ensembl_id' column (Geneformer's tokenizer "
                             "needs it)")
        if "n_counts" not in adata.obs.columns:
            adata.obs["n_counts"] = np.asarray(adata.X.sum(axis=1)).ravel().astype(np.float64)
        labels = adata.obs[col].astype(str)
        for ct in celltypes:
            mask = labels.isin(celltypes[ct]).to_numpy()
            if mask.any():
                parts[ct].append(adata[mask].copy())
        del adata
    return {ct: (p[0] if len(p) == 1
                 else ad.concat(p, join="inner", merge="same", index_unique="-"))
            for ct, p in parts.items() if p}


def filter_genes_by_prevalence(a):
    """Genes expressed in at least min(10, n_cells) cells, as extract_embeddings.py."""
    threshold = min(10, a.n_obs)
    keep = np.asarray((a.X > 0).sum(axis=0)).ravel() >= threshold
    return a[:, keep].copy()


def tokenize(a, workdir, nproc):
    """(input_ids per cell of `a`, in a's cell order, {token id: gene name}) through
    Geneformer's V2 tokenizer, as extract_embeddings.py runs it; names are
    adata.var_names (extract_embeddings.token_labels, var_index 'var_names')."""
    from datasets import load_from_disk
    from geneformer import TranscriptomeTokenizer
    from extract_embeddings import token_labels

    data_dir, tok_dir = Path(workdir) / "input", Path(workdir) / "tokenized"
    data_dir.mkdir(parents=True)
    tok_dir.mkdir(parents=True)
    a = a.copy()
    a.obs["__cell_id__"] = a.obs_names.astype(str)
    path = data_dir / "instance.h5ad"
    a.write_h5ad(path)
    tk = TranscriptomeTokenizer(custom_attr_name_dict={"__cell_id__": "__cell_id__"},
                                nproc=nproc, model_version="V2")
    tk.tokenize_data(data_directory=data_dir, output_directory=tok_dir, output_prefix="instance",
                     file_format="h5ad")
    data = load_from_disk(str(tok_dir / "instance.dataset"))
    by_id = dict(zip(data["__cell_id__"], data["input_ids"]))
    _, labels = token_labels(path, "var_names")
    return [by_id[c] for c in a.obs["__cell_id__"]], labels


def mask_positions(seqs, ratio, special, rng):
    """Per cell, the sorted positions of round(ratio * n) of its n gene tokens, drawn
    at random (special tokens are never masked)."""
    out = []
    for s in seqs:
        pos = np.array([i for i, t in enumerate(s) if t not in special])
        k = int(round(ratio * len(pos)))
        out.append(np.sort(rng.choice(pos, k, replace=False)) if k else pos[:0])
    return out


def batches(n, size):
    for s in range(0, n, size):
        yield slice(s, min(n, s + size))


def pad(seqs, pad_id):
    """(ids (B, S) long, attention (B, S) bool) for a list of token lists."""
    width = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
    return ids, torch.arange(width)[None] < torch.tensor([len(s) for s in seqs])[:, None]


# ── Model ─────────────────────────────────────────────────────────────────────

def run_layers(model, h, att, start=0, shift=None):
    """States at layers start..n of the encoder, from h (B, S, d) = the states at the
    output of layer `start` (0: the embedding output); att (B, S) bool. The encoder
    layers are called one at a time, as BertEncoder does, so a pass can start from
    an earlier pass's states: the layers below a shift do not see it. shift =
    (layer, vec, where) adds vec to the tokens marked in where (B, S) at that
    layer's output (layer >= start), i.e. to hidden_states[layer]."""
    layers = model.bert.encoder.layer
    if not 0 <= start <= len(layers) or (shift is not None and not start <= shift[0] <= len(layers)):
        raise ValueError(f"start {start} / shift layer: must be 0..{len(layers)}, shift >= start")
    ext = model.bert.get_extended_attention_mask(att.long(), att.shape)    # additive (B,1,1,S)

    def shifted(h):
        _, vec, where = shift
        return h + vec.to(h.dtype).view(1, 1, -1) * where.unsqueeze(-1).to(h.dtype)

    if shift is not None and shift[0] == start:
        h = shifted(h)
    states = [h]
    for i, layer in enumerate(layers, start=1):
        if i <= start:
            continue
        out = layer(h, attention_mask=ext)
        h = out[0] if isinstance(out, tuple) else out
        if shift is not None and shift[0] == i:
            h = shifted(h)
        states.append(h)
    return states


def check_layers(model, ids, att, tol=1e-4):
    """run_layers against the model's own forward (hidden_states), on one batch: the
    manual pass depends on transformers' BertLayer, so check it in each run's
    environment."""
    with torch.no_grad():
        ref = model.bert(input_ids=ids, attention_mask=att.long(),
                         output_hidden_states=True).hidden_states
        got = run_layers(model, ref[0], att)
    err = max(float((g - r)[att].abs().max()) for g, r in zip(got, ref))
    scale = float(ref[-1][att].abs().mean())
    if err > tol * max(scale, 1.0):
        raise RuntimeError(f"manual layer pass differs from the model's forward by {err:.2e} "
                           f"(scale {scale:.2e}); transformers' BertLayer changed?")
    return err


@torch.no_grad()
def gene_means(model, seqs, special, batch_size, device, pad_id):
    """(token ids (n,), (layers+1, n, d) float32): each gene's hidden state averaged
    over the cells whose input contains it, unmasked, for the genes in at least
    min(10, n_cells) inputs (extract_embeddings.gene_embeddings and its filter)."""
    ids_all = sorted({t for s in seqs for t in s if t not in special})
    rows = torch.full((model.config.vocab_size,), -1, dtype=torch.long, device=device)
    rows[torch.tensor(ids_all, device=device)] = torch.arange(len(ids_all), device=device)
    special_t = torch.tensor(sorted(special), device=device)
    dtype = torch.float32 if device.type == "mps" else torch.float64
    sums, counts = None, torch.zeros(len(ids_all), dtype=dtype, device=device)
    for s in batches(len(seqs), batch_size):
        ids, att = (x.to(device) for x in pad(seqs[s], pad_id))
        hidden = model.bert(input_ids=ids, attention_mask=att.long(),
                            output_hidden_states=True).hidden_states
        if sums is None:
            sums = [torch.zeros(len(ids_all), hidden[0].shape[-1], dtype=dtype, device=device)
                    for _ in hidden]
        genes = att & ~torch.isin(ids, special_t)
        r = rows[ids[genes]]
        counts.index_add_(0, r, torch.ones(len(r), dtype=dtype, device=device))
        for L, h in enumerate(hidden):
            sums[L].index_add_(0, r, h[genes].to(dtype))
    keep = counts >= min(10, len(seqs))
    emb = torch.stack([(x[keep] / counts[keep, None]).float().cpu() for x in sums])
    return torch.tensor(ids_all)[keep.cpu()], emb


def rank_readout(logp, values):
    """For one cell: logp (|P|, |G|) = log P(held-out gene g at masked position p),
    values (|P|,) the positions' relative ranks; returns each gene's predicted
    relative rank (|G|,): sum_p P * v_p / sum_p P."""
    w = torch.softmax(logp, dim=0)          # = P / sum_p P, computed stably in log space
    return (w * values[:, None]).sum(0)


def readout(model, h, ids, genes, is_masked, masked, acc):
    """Add one batch's read-outs to acc (6, vocab): per masked gene, (1, true relative
    rank, pred_rank, p_correct, logp_correct, top1), from the last-layer states h."""
    logp_all = torch.log_softmax(model.cls(h[is_masked]).float(), dim=-1)    # (sum |P|, V)
    start = 0
    for i, P in enumerate(masked):
        k = len(P)
        if not k:
            continue
        lp = logp_all[start:start + k]
        start += k
        P = torch.as_tensor(P, dtype=torch.long, device=h.device)
        G = ids[i, P]
        n = int(genes[i].sum())
        values = 1.0 - (P - 1).float() / max(n - 1, 1)        # <cls> at position 0
        cols = lp[:, G]                                        # (|P|, |G|)
        diag = cols.diagonal()
        for row, x in enumerate((torch.ones_like(values), values, rank_readout(cols, values),
                                 diag.exp(), diag, (lp.argmax(1) == G).float())):
            acc[row].index_add_(0, G, x.float())


@torch.no_grad()
def predict_all(model, seqs, masked, conditions, batch_size, device, special, pad_id, mask_id,
                cls_rows=None):
    """Every condition on the same masked positions: (c, 6, vocab) float32 sums over
    the cells where each gene was masked of (1, true relative rank, pred_rank,
    p_correct, logp_correct, top1). conditions: [(shift or None, shift_masked)];
    shift = (layer, vec) adds vec at that layer's output to the unmasked gene tokens,
    and to the masked ones if shift_masked. Each batch runs the unshifted pass once,
    keeping every layer's states; a condition shifted at layer L starts from the
    states at L. With cls_rows (indices into the cells), also returns the final <cls>
    output (position 0) of those cells per condition, (c, rows, d) float16 (masked
    tokens are seen by <cls> here, so dir passes have their own)."""
    acc = torch.zeros(len(conditions), 6, model.config.vocab_size, device=device)
    special_t = torch.tensor(sorted(special), device=device)
    rows = [] if cls_rows is None else sorted(int(r) for r in cls_rows)
    where_row = {r: i for i, r in enumerate(rows)}
    cls = torch.zeros(len(conditions), len(rows), model.config.hidden_size, dtype=torch.float16)
    for s in batches(len(seqs), batch_size):
        keep = [(i, where_row[r]) for i, r in enumerate(range(s.start, s.stop)) if r in where_row]
        ids, att = (x.to(device) for x in pad(seqs[s], pad_id))
        is_masked = torch.zeros_like(att)
        for i, P in enumerate(masked[s]):
            is_masked[i, torch.as_tensor(P, dtype=torch.long)] = True
        genes = att & ~torch.isin(ids, special_t)
        states = run_layers(model, model.bert.embeddings(input_ids=ids.masked_fill(is_masked,
                                                                                    mask_id)), att)
        for c, (shift, shift_masked) in enumerate(conditions):
            if shift is None:
                h = states[-1]
            else:
                L, vec = shift
                where = (genes & ~is_masked) | (is_masked if shift_masked else
                                                 torch.zeros_like(is_masked))
                h = run_layers(model, states[L], att, start=L,
                               shift=(L, vec.to(device), where))[-1]
            readout(model, h, ids, genes, is_masked, masked[s], acc[c])
            if keep:
                bi, ci = zip(*keep)
                cls[c, list(ci)] = h[list(bi), 0].half().cpu()
    return acc.cpu() if cls_rows is None else (acc.cpu(), cls)


def predict(model, seqs, masked, batch_size, device, special, pad_id, mask_id, shift=None,
            shift_masked=False):
    """One condition: (6, vocab), as predict_all."""
    return predict_all(model, seqs, masked, [(shift, shift_masked)], batch_size, device, special,
                       pad_id, mask_id)[0]


def paired_shift(a, b):
    """((layers+1, d) shift from instance a to b, number of shared genes): the mean
    over the genes both have of b's gene embedding minus a's; a, b = gene_means
    results."""
    shared, ia, ib = np.intersect1d(a[0].numpy(), b[0].numpy(), return_indices=True)
    if not len(shared):
        raise ValueError("the two instances share no genes")
    return (b[1][:, ib] - a[1][:, ia]).mean(1), len(shared)


# ── Shifts (as in scgpt_embeds' offset_intervention.py) ───────────────────────

def pair_shifts(keys, emb, batch_pairs, ct_pairs):
    """Paired shifts and their held-out means; keys are (batch, celltype), emb =
    {key: gene_means result}. Returns a dict of four dicts:
      chem_own      {(A, B, k): (shift, n shared genes)}
      chem_heldout  {(A, B, k): (mean of chem_own over the other celltypes, n)}
      ct_own        {(X, a, b): (shift, n shared genes)}
      ct_heldout    {(X, a, b): (mean of ct_own over the other batches, n)}"""
    have = set(keys)
    # dict.fromkeys, not a set: a fixed order, so the seeded random shifts go to
    # the same instances in every run (set order changes with the hash seed)
    cts = list(dict.fromkeys(c for _, c in keys))
    bts = list(dict.fromkeys(b for b, _ in keys))
    chem = {(a, b, ct): paired_shift(emb[(a, ct)], emb[(b, ct)])
            for a, b in batch_pairs for ct in cts if (a, ct) in have and (b, ct) in have}
    cell = {(x, a, b): paired_shift(emb[(x, a)], emb[(x, b)])
            for a, b in ct_pairs for x in bts if (x, a) in have and (x, b) in have}

    def held_out(own, same_group, label):
        out = {}
        for key in own:
            others = [own[o][0] for o in own if o != key and same_group(o, key)]
            if others:
                out[key] = (torch.stack(others).mean(0), len(others))
            else:
                print(f"  no held-out shift for {key}: no other {label}", flush=True)
        return out

    return {"chem_own": chem,
            "chem_heldout": held_out(chem, lambda o, k: o[:2] == k[:2], "celltype"),
            "ct_own": cell,
            "ct_heldout": held_out(cell, lambda o, k: o[1:] == k[1:], "batch")}


def plan_shifts(keys, shifts, doses, n_random, seed):
    """{instance key: [(name, (layers+1, d) shift)]} from pair_shifts' result."""
    rng = np.random.default_rng(seed)

    def random_like(D, n):
        out = []
        for _ in range(n):
            r = torch.from_numpy(rng.normal(size=D.shape)).float()
            out.append(r * (D.norm(dim=1, keepdim=True) / r.norm(dim=1, keepdim=True)))
        return out

    plan = {k: [] for k in keys}
    for (a, b, ct), (own, _) in shifts["chem_own"].items():
        tag = f"{a}>{b}"
        held = shifts["chem_heldout"].get((a, b, ct), (None,))[0]
        ref = own if held is None else held
        plan[(a, ct)].append((f"chem_own|{tag}", own))
        plan[(b, ct)].append((f"chem_own_rev|{tag}", -own))
        if held is not None:
            plan[(a, ct)].append((f"chem_heldout|{tag}", held))
            plan[(b, ct)].append((f"chem_heldout_rev|{tag}", -held))
            for f in doses:
                plan[(a, ct)].append((f"chem_heldout_x{f:g}|{tag}", f * held))
                plan[(b, ct)].append((f"chem_heldout_rev_x{f:g}|{tag}", -f * held))
        for j, R in enumerate(random_like(ref, n_random)):
            plan[(a, ct)].append((f"rand_chem{j}|{tag}", R))
        for j, R in enumerate(random_like(ref, n_random)):
            plan[(b, ct)].append((f"rand_chem{j}_rev|{tag}", R))
    for (x, a, b), (own, _) in shifts["ct_own"].items():
        tag = f"{a}>{b}"
        held = shifts["ct_heldout"].get((x, a, b), (None,))[0]
        plan[(x, a)].append((f"ct_own|{tag}", own))
        if held is not None:
            plan[(x, a)].append((f"ct_heldout|{tag}", held))
            for f in doses:
                plan[(x, a)].append((f"ct_heldout_x{f:g}|{tag}", f * held))
        for j, R in enumerate(random_like(own if held is None else held, n_random)):
            plan[(x, a)].append((f"rand_ct{j}|{tag}", R))
    return plan


def stack_shifts(shifts, width):
    """{name: array} for shifts.npz: <kind>_keys, <kind> (pairs, layers+1, d) and the
    counts (<kind>_n_shared for own shifts, <kind>_n for held-out means)."""
    out = {}
    for kind, d in shifts.items():
        out[f"{kind}_keys"] = np.array(["|".join(k) for k in d], dtype=str)
        out[kind] = (torch.stack([D for D, _ in d.values()]).numpy() if d
                     else np.zeros((0, *width), np.float32))
        out[f"{kind}_n_shared" if kind.endswith("_own") else f"{kind}_n"] = \
            np.array([n for _, n in d.values()], int)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_pairs(items):
    out = []
    for it in items:
        a, sep, b = it.partition(":")
        if not sep or not a or not b:
            raise ValueError(f"expected A:B, got {it!r}")
        out.append((a, b))
    return out


def safe(name):
    return name.replace(" ", "_").replace("/", "-")


def instance_seed(seed, key):
    """A seed of its own per instance (batch, celltype), the same in every job."""
    return zlib.crc32(f"{seed}|{key[0]}|{key[1]}".encode())


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["all", "shifts", "eval", "merge"], default="all",
                   help="all: everything in this job; or shifts (one job), eval (one job "
                        "per task: an instance's chunk of cells), merge (CPU)")
    p.add_argument("--task", type=int, default=None,
                   help="eval: the task index (default $SLURM_ARRAY_TASK_ID); tasks past "
                        "the last exit")
    p.add_argument("--chunk-size", type=int, default=1000, help="cells per eval task")
    p.add_argument("--cls-cells", type=int, default=1000,
                   help="per instance, the cells whose final <cls> output is saved for "
                        "every condition (a random subset; 0: none)")
    p.add_argument("--data-dir")
    p.add_argument("--batch", action="append", default=[], metavar="NAME=PATTERN[,PATTERN]",
                   help="a batch made of the h5ad files matching the patterns (names "
                        "without .h5ad, globs allowed), pooled; repeatable. Any other "
                        "batch name is the file of that name")
    p.add_argument("--batch-pairs", nargs="+", default=[],
                   help="batch pairs A:B; the chemistry shift moves A to B")
    p.add_argument("--datasets", nargs="+", default=[],
                   help="further batches (default: those of --batch-pairs)")
    p.add_argument("--celltypes", nargs="+", default=[],
                   help="NAME, or NAME=LABEL1,LABEL2 to pool differently named cells")
    p.add_argument("--celltype-col", default="celltype_coarse")
    p.add_argument("--celltype-pairs", nargs="+", default=[], help="a:b, the shift moves a to b")
    p.add_argument("--layers", type=int, nargs="+", default=[],
                   help="layers whose output gets the shift (0 = embedding output .. n)")
    p.add_argument("--max-cells", type=int, default=0,
                   help="per instance, drawn at random (0: every cell)")
    p.add_argument("--min-cells", type=int, default=20)
    p.add_argument("--heldout-ratio", type=float, default=0.15)
    p.add_argument("--doses", type=float, nargs="*", default=[0.5, 2.0])
    p.add_argument("--n-random", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--nproc", type=int, default=4, help="tokenizer processes")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--model-dir",
                   help="the pretrained BertForMaskedLM (model.safetensors, config.json)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    if args.stage != "merge":
        missing = [f"--{k.replace('_', '-')}" for k in ("data_dir", "model_dir")
                   if getattr(args, k) is None] + (["--celltypes"] if not args.celltypes else [])
        if missing:
            p.error(f"--stage {args.stage} needs {' '.join(missing)}")
    return args


def setup(args):
    """The model and everything the stages share, as a namespace."""
    from geneformer import TOKEN_DICTIONARY_FILE
    from geneformer import perturber_utils as pu

    device = torch.device(args.device if args.device != "auto"
                          else "cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    with open(TOKEN_DICTIONARY_FILE, "rb") as f:
        token_dict = pickle.load(f)
    e = SimpleNamespace(device=device)
    e.special = {v for k, v in token_dict.items() if k.startswith("<")}
    e.pad_id, e.mask_id = token_dict["<pad>"], token_dict["<mask>"]
    e.model = pu.load_model("Pretrained", 0, str(args.model_dir), mode="eval").to(device)
    e.n_layers = len(e.model.bert.encoder.layer)
    bad = [L for L in args.layers if not 0 <= L <= e.n_layers]
    if bad:
        raise ValueError(f"--layers {bad}: shifts go at layers 0..{e.n_layers}")
    print(f"device {device}; {e.n_layers} layers; held-out ratio {args.heldout_ratio}",
          flush=True)
    return e


def run_names(args):
    names = list(dict.fromkeys([d for pr in parse_pairs(args.batch_pairs) for d in pr]
                               + args.datasets))
    if not names:
        raise ValueError("no batches: give --batch-pairs or --datasets")
    return names


def load_instances(args, e, only=None):
    """(keys, {key: input_ids per cell}, {token id: gene name}) for every (batch,
    celltype) of the run with at least --min-cells cells, or only for the key `only`."""
    names = run_names(args) if only is None else [only[0]]
    files = resolve_batches(args.batch, names, args.data_dir)
    keys, seqs, labels = [], {}, {}
    with tempfile.TemporaryDirectory(prefix="gf_offset_") as tmp:
        for b in names:
            labels_of = celltype_labels(args.celltypes)
            cts = list(labels_of) if only is None else [only[1]]
            print(f"{b}: {len(files[b])} file(s)", flush=True)
            pooled = pool_celltypes(files[b], {c: labels_of[c] for c in cts}, args.celltype_col)
            for ct in cts:
                if ct not in pooled:
                    print(f"  {b} / {ct}: not in the batch, skipped", flush=True)
                    continue
                a = filter_genes_by_prevalence(pooled.pop(ct))
                if args.max_cells and a.n_obs > args.max_cells:
                    rng = np.random.default_rng(instance_seed(args.seed, (b, ct)))
                    a = a[np.sort(rng.choice(a.n_obs, args.max_cells, replace=False))].copy()
                if a.n_obs < args.min_cells:
                    print(f"  {b} / {ct}: {a.n_obs} cells < --min-cells, skipped", flush=True)
                    continue
                s, lab = tokenize(a, Path(tmp) / f"{safe(b)}__{safe(ct)}", args.nproc)
                labels.update(lab)
                seqs[(b, ct)] = s
                keys.append((b, ct))
                lengths = [sum(t not in e.special for t in x) for x in s]
                print(f"  {b} / {ct}: {len(s)} cells; median {np.median(lengths):.0f} genes "
                      "per input", flush=True)
            del pooled
    return keys, seqs, labels


def stage_shifts(args, e, out):
    """The estimation pass over every cell of every instance: shifts.npz, plan.pt (the
    shifts each instance gets, random ones included), tasks.tsv (one eval task per
    instance and chunk of cells) and labels.json (token id -> gene name)."""
    keys, seqs, labels = load_instances(args, e)
    if not keys:
        raise SystemExit("error: no instance has enough cells")
    emb = {k: gene_means(e.model, seqs[k], e.special, args.batch_size, e.device, e.pad_id)
           for k in keys}
    shifts = pair_shifts(keys, emb, parse_pairs(args.batch_pairs), parse_pairs(args.celltype_pairs))
    E = emb[keys[0]][1]
    np.savez(out / "shifts.npz", keys=np.array(["|".join(k) for k in keys]),
             n_cells=np.array([len(seqs[k]) for k in keys]),
             n_genes=np.array([len(emb[k][0]) for k in keys]),
             n_genes_median=np.array([np.median([sum(t not in e.special for t in x)
                                                 for x in seqs[k]]) for k in keys]),
             layers=np.array(args.layers), heldout_ratio=np.array(args.heldout_ratio),
             **stack_shifts(shifts, (E.shape[0], E.shape[2])))
    plan = plan_shifts(keys, shifts, args.doses, args.n_random, args.seed)
    torch.save({"|".join(k): v for k, v in plan.items()}, out / "plan.pt")
    with open(out / "labels.json", "w") as f:
        json.dump({str(k): str(v) for k, v in labels.items()}, f)
    with open(out / "tasks.tsv", "w") as f:
        f.write("task\tbatch\tcelltype\tstart\tstop\n")
        t = 0
        for k in keys:
            n = len(seqs[k])
            for start in range(0, n, args.chunk_size):
                f.write(f"{t}\t{k[0]}\t{k[1]}\t{start}\t{min(n, start + args.chunk_size)}\n")
                t += 1
    print(f"shifts of {len(keys)} instances; {t} eval tasks -> {out / 'tasks.tsv'}", flush=True)
    return t


def read_tasks(out):
    with open(out / "tasks.tsv") as f:
        next(f)
        return [(int(t), b, c, int(a), int(z)) for t, b, c, a, z in
                (line.rstrip("\n").split("\t") for line in f)]


def stage_eval(args, e, out, task):
    """One instance's chunk of cells: base and every condition of its plan; writes
    parts/<batch>__<celltype>__<start>.npz."""
    _, b, ct, start, stop = read_tasks(out)[task]
    plan = torch.load(out / "plan.pt")["|".join((b, ct))]
    keys, seqs, _ = load_instances(args, e, only=(b, ct))
    if keys != [(b, ct)] or len(seqs[(b, ct)]) < stop:
        raise RuntimeError(f"{b} / {ct} prepared differently from the shifts stage")
    s_all = seqs[(b, ct)]
    seed = instance_seed(args.seed, (b, ct))
    masked = mask_positions(s_all, args.heldout_ratio, e.special, np.random.default_rng(seed + 1))
    n_all = len(s_all)
    cls_cells = np.sort(np.random.default_rng(seed + 2).choice(
        n_all, min(args.cls_cells, n_all), replace=False)) if args.cls_cells else np.array([], int)
    chunk, masked = s_all[start:stop], masked[start:stop]
    rows = cls_cells[(cls_cells >= start) & (cls_cells < stop)]
    ids, att = (x.to(e.device) for x in pad(chunk[:args.batch_size], e.pad_id))
    print(f"  manual layer pass vs the model's forward: max |diff| "
          f"{check_layers(e.model, ids, att):.1e}", flush=True)
    conds, todo = ["base"], [(None, False)]
    for L in args.layers:
        for name, D in plan:
            if L < e.n_layers:
                conds.append(f"L{L}|ctx|{name}")
                todo.append(((L, D[L]), False))
            if L >= 1:
                conds.append(f"L{L}|dir|{name}")
                todo.append(((L, D[L]), True))
    A, cls = predict_all(e.model, chunk, masked, todo, args.batch_size, e.device, e.special,
                         e.pad_id, e.mask_id, cls_rows=rows - start)
    hit = torch.nonzero(A[0, 0]).ravel()
    (out / "parts").mkdir(exist_ok=True)
    np.savez(out / "parts" / f"{safe(b)}__{safe(ct)}__{start}.npz",
             ids=hit.numpy(), sums=A[:, :, hit].numpy(), conditions=np.array(conds),
             cls=cls.numpy(), cls_cells=rows)
    print(f"  {b} / {ct} cells {start}-{stop}: {len(conds)} conditions, {len(hit)} genes "
          f"masked, <cls> of {len(rows)} cells", flush=True)


def stage_merge(out):
    """Sum each instance's parts into <batch>__<celltype>.npz: genes (n,), token_ids,
    n_heldout, target_sum (n,), conditions (c,), pred_rank, p_correct, logp_correct,
    top1 (c, n), cls_conditions (c,), cls (c, cells, d) float16 and cls_cells (cells,)
    their indices in the instance."""
    with open(out / "labels.json") as f:
        labels = {int(k): v for k, v in json.load(f).items()}
    by_inst = {}
    for _, b, ct, start, _ in read_tasks(out):
        by_inst.setdefault((b, ct), []).append(out / "parts" / f"{safe(b)}__{safe(ct)}__{start}.npz")
    for (b, ct), paths in by_inst.items():
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            raise SystemExit(f"error: {b} / {ct}: missing parts {missing}; rerun those tasks")
        parts = [np.load(p) for p in paths]
        conds = parts[0]["conditions"]
        ids = np.unique(np.concatenate([z["ids"] for z in parts]))
        col = {int(i): j for j, i in enumerate(ids)}
        A = np.zeros((len(conds), 6, len(ids)))
        for z in parts:
            if not np.array_equal(z["conditions"], conds):
                raise RuntimeError(f"{b} / {ct}: parts with different conditions")
            A[:, :, [col[int(i)] for i in z["ids"]]] += z["sums"]
        np.savez(out / f"{safe(b)}__{safe(ct)}.npz",
                 genes=np.array([labels.get(int(t), str(t)) for t in ids]), token_ids=ids,
                 n_heldout=A[0, 0], target_sum=A[0, 1], conditions=conds,
                 pred_rank=A[:, 2], p_correct=A[:, 3], logp_correct=A[:, 4], top1=A[:, 5],
                 cls_conditions=conds,
                 cls=np.concatenate([z["cls"] for z in parts], axis=1),
                 cls_cells=np.concatenate([z["cls_cells"] for z in parts]))
        print(f"  {b} / {ct}: {len(paths)} part(s), {len(ids)} genes", flush=True)
    print(f"merged {len(by_inst)} instances -> {out}")


def main(argv=None):
    args = parse_args(argv)
    out = Path(args.out)
    if args.stage == "merge":
        stage_merge(out)
        return
    if args.stage in ("all", "shifts"):
        if out.exists() and any(out.glob("*.npz")):
            raise SystemExit(f"error: {out} already holds results; give a new --out")
        out.mkdir(parents=True, exist_ok=True)
    else:
        task = args.task if args.task is not None else int(os.environ.get("SLURM_ARRAY_TASK_ID", -1))
        if task < 0:
            raise SystemExit("error: --stage eval needs --task or $SLURM_ARRAY_TASK_ID")
        if task >= len(read_tasks(out)):            # before loading the model
            print(f"task {task}: past the last ({len(read_tasks(out)) - 1}), nothing to do")
            return
    e = setup(args)
    if args.stage in ("all", "shifts"):
        n_tasks = stage_shifts(args, e, out)
        if args.stage == "shifts":
            return
        for task in range(n_tasks):
            stage_eval(args, e, out, task)
        stage_merge(out)
        return
    stage_eval(args, e, out, task)


if __name__ == "__main__":
    main()
