"""Tests for scripts/offset_intervention.py on a tiny random BertForMaskedLM with
Geneformer's vocabulary: the manual layer pass and its shift against hidden_states,
restarting from kept states, the rank read-out,
the batched read-outs against the model's own forward cell by cell, the gene means,
and a run on synthetic h5ad files through Geneformer's tokenizer.

    python -m pytest tests/test_offset_intervention.py   (in an env with geneformer)
"""

import pickle
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT)]

import offset_intervention as oi  # noqa: E402
from geneformer import GENE_MEDIAN_FILE, TOKEN_DICTIONARY_FILE  # noqa: E402

CPU = torch.device("cpu")


@pytest.fixture(scope="module")
def token_dict():
    with open(TOKEN_DICTIONARY_FILE, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def special(token_dict):
    return {v for k, v in token_dict.items() if k.startswith("<")}


@pytest.fixture(scope="module")
def model_dir(tmp_path_factory, token_dict):
    from transformers import BertConfig, BertForMaskedLM
    torch.manual_seed(0)
    cfg = BertConfig(hidden_size=16, num_hidden_layers=2, num_attention_heads=2,
                     intermediate_size=32, vocab_size=max(token_dict.values()) + 1,
                     max_position_embeddings=128, hidden_act="relu")
    d = tmp_path_factory.mktemp("model")
    BertForMaskedLM(cfg).save_pretrained(d)
    return d


@pytest.fixture(scope="module")
def model(model_dir):
    from geneformer import perturber_utils as pu
    return pu.load_model("Pretrained", 0, str(model_dir), mode="eval").to(CPU)


@pytest.fixture(scope="module")
def seqs(token_dict):
    """3 cells: <cls>, 10 / 7 / 4 genes, <eos>."""
    genes = [v for k, v in token_dict.items() if not k.startswith("<")][:20]
    cls, eos = token_dict["<cls>"], token_dict["<eos>"]
    return [[cls] + genes[c:c + n] + [eos] for c, n in enumerate((10, 7, 4))]


def test_run_layers_reproduces_and_shifts_hidden_states(model, seqs, token_dict):
    ids, att = oi.pad(seqs, token_dict["<pad>"])
    assert oi.check_layers(model, ids, att) < 1e-5
    vec = torch.randn(16)
    where = att.clone()
    where[:, 0] = False
    with torch.no_grad():
        base = model.bert(input_ids=ids, attention_mask=att.long(),
                          output_hidden_states=True).hidden_states
        for L in (0, 1, 2):
            got = oi.run_layers(model, base[0], att, shift=(L, vec, where))
            m = att
            assert torch.allclose(got[L][m], (base[L] + vec * where.unsqueeze(-1))[m], atol=1e-5)
            for j in range(L):
                assert torch.allclose(got[j][m], base[j][m], atol=1e-5)
            if L < 2:
                assert not torch.allclose(got[L + 1][m], base[L + 1][m], atol=1e-4)
            part = oi.run_layers(model, base[L], att, start=L, shift=(L, vec, where))
            assert torch.allclose(part[-1][m], got[-1][m], atol=1e-5)
    with pytest.raises(ValueError):
        oi.run_layers(model, base[0], att, start=2, shift=(1, vec, where))


def test_rank_readout_is_the_probability_weighted_rank():
    v = torch.tensor([0.9, 0.5, 0.1], dtype=torch.float64)
    for p, want in (([0.02, 0.06, 0.02], 0.5), ([0.06, 0.03, 0.01], 0.7)):
        got = oi.rank_readout(torch.tensor(p, dtype=torch.float64).log()[:, None], v)
        assert torch.allclose(got, torch.tensor([want], dtype=torch.float64))


def _by_hand(model, seq, P, token_dict):
    """Per masked gene of one cell: (value, pred_rank, p_correct, top1) from the
    model's own forward on the masked cell alone."""
    inp = torch.tensor(seq)
    inp[P] = token_dict["<mask>"]
    with torch.no_grad():
        logp = torch.log_softmax(model(input_ids=inp[None]).logits[0].double(), -1)
    n = len(seq) - 2
    v = 1.0 - (torch.tensor(P) - 1).double() / (n - 1)
    G = torch.tensor(seq)[P]
    prob = logp[P][:, G].exp()
    pred = (prob * v[:, None]).sum(0) / prob.sum(0)
    return G, v, pred, prob.diagonal(), (logp[P].argmax(1) == G).double()


def test_predict_matches_each_cell_alone(model, seqs, token_dict, special):
    masked = oi.mask_positions(seqs, 0.3, special, np.random.default_rng(0))
    run = dict(model=model, seqs=seqs, masked=masked, device=CPU,
               special=special, pad_id=token_dict["<pad>"], mask_id=token_dict["<mask>"])
    acc = oi.predict(batch_size=2, **run)
    assert acc.dtype == torch.float32
    assert torch.allclose(acc, oi.predict(batch_size=1, **run), atol=1e-5)
    acc = acc.double()
    want = torch.zeros_like(acc)
    for seq, P in zip(seqs, masked):
        G, v, pred, pc, top1 = _by_hand(model, seq, list(P), token_dict)
        for row, x in enumerate((torch.ones_like(v), v, pred, pc, pc.log(), top1)):
            want[row].index_add_(0, G, x)
    assert torch.allclose(acc, want, atol=1e-4)
    assert int(acc[0].sum()) == sum(len(P) for P in masked) == 3 + 2 + 1


def test_shift_edges(model, seqs, token_dict, special):
    masked = oi.mask_positions(seqs, 0.3, special, np.random.default_rng(0))
    run = dict(model=model, seqs=seqs, masked=masked, batch_size=2, device=CPU,
               special=special, pad_id=token_dict["<pad>"], mask_id=token_dict["<mask>"])
    base = oi.predict(**run)
    for L, m in ((0, False), (1, True), (2, True)):
        assert torch.allclose(oi.predict(**run, shift=(L, torch.zeros(16)), shift_masked=m), base)
    vec = torch.randn(16) * 3
    # unmasked tokens shifted after the last layer: the head never reads them
    assert torch.allclose(oi.predict(**run, shift=(2, vec), shift_masked=False), base, atol=1e-6)
    assert not torch.allclose(oi.predict(**run, shift=(2, vec), shift_masked=True)[2], base[2])
    ctx = oi.predict(**run, shift=(1, vec), shift_masked=False)
    assert not torch.allclose(ctx[2], base[2])
    assert torch.equal(ctx[:2], base[:2])                       # same genes, same targets


def test_predict_all_equals_each_condition_alone(model, seqs, token_dict, special):
    masked = oi.mask_positions(seqs, 0.3, special, np.random.default_rng(0))
    kw = dict(batch_size=2, device=CPU, special=special, pad_id=token_dict["<pad>"],
              mask_id=token_dict["<mask>"])
    a, b = torch.randn(16), torch.randn(16)
    conds = [(None, False), ((0, a), False), ((1, b), True), ((2, a), True), ((1, a), False)]
    A = oi.predict_all(model, seqs, masked, conds, **kw)
    for c, (shift, m) in enumerate(conds):
        one = oi.predict(model, seqs, masked, shift=shift, shift_masked=m, **kw)
        assert torch.allclose(A[c], one, atol=1e-5)


def test_mask_positions_skip_special_tokens(seqs, special):
    a = oi.mask_positions(seqs, 0.3, special, np.random.default_rng(1))
    b = oi.mask_positions(seqs, 0.3, special, np.random.default_rng(1))
    for s, P, Q in zip(seqs, a, b):
        assert np.array_equal(P, Q)
        assert len(P) == round(0.3 * (len(s) - 2))
        assert all(s[p] not in special for p in P)


def test_gene_means_average_each_gene_over_its_inputs(model, seqs, token_dict, special):
    ids, emb = oi.gene_means(model, seqs, special, 2, CPU, token_dict["<pad>"])
    full_ids, att = oi.pad(seqs, token_dict["<pad>"])
    with torch.no_grad():
        hidden = model.bert(input_ids=full_ids, attention_mask=att.long(),
                            output_hidden_states=True).hidden_states
    # min(10, 3 cells) = 3: only genes in all three inputs are kept
    in_all = sorted(set(seqs[0][1:-1]) & set(seqs[1][1:-1]) & set(seqs[2][1:-1]))
    assert ids.tolist() == in_all
    for L, h in enumerate(hidden):
        for j, g in enumerate(ids.tolist()):
            assert torch.allclose(emb[L, j], h[(full_ids == g) & att].mean(0), atol=1e-5)


def test_pair_and_plan_shifts():
    keys = [(b, c) for b in ("3p", "5p") for c in ("T", "B", "M")]
    g = torch.Generator().manual_seed(0)
    emb = {k: (torch.arange(i % 3, i % 3 + 5), torch.randn(3, 5, 4, generator=g))
           for i, k in enumerate(keys)}
    s = oi.pair_shifts(keys, emb, [("3p", "5p")], [("T", "B")])
    D = lambda x, y: oi.paired_shift(emb[x], emb[y])[0]
    held, n = s["chem_heldout"][("3p", "5p", "T")]
    assert n == 2 and torch.allclose(held, (D(("3p", "B"), ("5p", "B")) +
                                            D(("3p", "M"), ("5p", "M"))) / 2)
    assert torch.allclose(s["ct_heldout"][("3p", "T", "B")][0], D(("5p", "T"), ("5p", "B")))
    plan = {k: dict(v) for k, v in oi.plan_shifts(keys, s, [2.0], 1, 0).items()}
    assert torch.allclose(plan[("3p", "T")]["chem_heldout_x2|3p>5p"], 2 * held)
    ct = s["ct_heldout"][("3p", "T", "B")][0]
    assert torch.allclose(plan[("3p", "T")]["ct_heldout_x2|T>B"], 2 * ct)
    assert torch.allclose(plan[("5p", "T")]["chem_heldout_rev|3p>5p"], -held)
    assert torch.allclose(plan[("5p", "T")]["chem_heldout_rev_x2|3p>5p"], -2 * held)


def test_main_on_synthetic_sites(tmp_path, model_dir, token_dict):
    with open(GENE_MEDIAN_FILE, "rb") as f:
        medians = pickle.load(f)
    ens = [k for k in token_dict if not k.startswith("<") and k in medians][:40]
    rng = np.random.default_rng(1)
    for site in ("s1", "s2"):
        for chem, s in (("3p", 0.0), ("5p", 1.0)):
            X = rng.poisson(1.5 + s * (np.arange(40) % 3 == 0), size=(50, 40)).astype(np.float32)
            a = ad.AnnData(X)
            a.var_names = [f"G{i}" for i in range(40)]
            a.var["ensembl_id"] = ens
            a.obs_names = [f"{site}_{chem}_{i}" for i in range(50)]
            a.obs["celltype_coarse"] = ["T cell"] * 25 + ["B cell"] * 15 + ["NK cell"] * 10
            if site == "s1":
                a.obs["n_counts"] = X.sum(1)
            a.write_h5ad(tmp_path / f"{site}_{chem}.h5ad")
    out = tmp_path / "out"
    oi.main(["--data-dir", str(tmp_path), "--batch", "3p=*_3p", "--batch", "5p=*_5p",
             "--batch-pairs", "3p:5p", "--celltypes", "T cell", "B cell", "NK cell",
             "--celltype-pairs", "T cell:B cell", "--layers", "0", "1", "2", "--n-random", "1",
             "--doses", "2", "--nproc", "1", "--device", "cpu", "--model-dir", str(model_dir),
             "--out", str(out)])
    s = np.load(out / "shifts.npz")
    assert list(s["n_cells"]) == [50, 30, 20] * 2                    # sites pooled
    assert list(s["chem_own_keys"]) == [f"3p|5p|{c}" for c in ("T cell", "B cell", "NK cell")]
    assert s["chem_own"].shape == (3, 3, 16) and list(s["chem_heldout_n"]) == [2, 2, 2]
    assert np.allclose(s["chem_heldout"][0], (s["chem_own"][1] + s["chem_own"][2]) / 2, atol=1e-6)
    r = np.load(out / "3p__T_cell.npz")
    conds = list(r["conditions"])
    assert conds[0] == "base"
    assert {"L0|ctx|chem_heldout|3p>5p", "L1|dir|chem_own|3p>5p", "L2|dir|rand_chem0|3p>5p",
            "L1|ctx|ct_heldout|T cell>B cell"} <= set(conds)
    assert not any(c.startswith("L0|dir") or c.startswith("L2|ctx") for c in conds)
    for key in ("pred_rank", "p_correct", "logp_correct", "top1"):
        assert r[key].shape == (len(conds), len(r["genes"]))
    assert set(r["genes"]) <= {f"G{i}" for i in range(40)}
    assert (r["n_heldout"] > 0).all()
    pr = r["pred_rank"] / r["n_heldout"]
    assert ((pr >= 0) & (pr <= 1)).all()
    assert r["cls"].shape == (len(conds), 50, 16) and list(r["cls_conditions"]) == conds
    with pytest.raises(SystemExit):                                  # never overwrites
        oi.main(["--data-dir", str(tmp_path), "--batch-pairs", "s1_3p:s1_5p", "--celltypes",
                 "T cell", "--layers", "1", "--model-dir", str(model_dir), "--out", str(out)])


def test_staged_run_equals_one_job(tmp_path, model_dir, token_dict):
    with open(GENE_MEDIAN_FILE, "rb") as f:
        medians = pickle.load(f)
    ens = [k for k in token_dict if not k.startswith("<") and k in medians][:40]
    rng = np.random.default_rng(2)
    for chem, sh in (("3p", 0.0), ("5p", 1.0)):
        X = rng.poisson(1.5 + sh * (np.arange(40) % 3 == 0), size=(45, 40)).astype(np.float32)
        a = ad.AnnData(X)
        a.var_names = [f"G{i}" for i in range(40)]
        a.var["ensembl_id"] = ens
        a.obs_names = [f"{chem}_{i}" for i in range(45)]
        a.obs["celltype_coarse"] = ["T cell"] * 25 + ["B cell"] * 20
        a.write_h5ad(tmp_path / f"s1_{chem}.h5ad")
    run = ["--data-dir", str(tmp_path), "--batch-pairs", "s1_3p:s1_5p", "--celltypes", "T cell",
           "B cell", "--celltype-pairs", "T cell:B cell", "--layers", "0", "2", "--n-random", "1",
           "--doses", "2", "--nproc", "1", "--device", "cpu", "--model-dir", str(model_dir),
           "--cls-cells", "12", "--chunk-size", "10"]
    one, staged = tmp_path / "one", tmp_path / "staged"
    oi.main(run + ["--out", str(one)])
    oi.main(run + ["--out", str(staged), "--stage", "shifts"])
    tasks = oi.read_tasks(staged)
    assert len(tasks) == 3 + 2 + 3 + 2
    for t in range(len(tasks) + 1):
        oi.main(run + ["--out", str(staged), "--stage", "eval", "--task", str(t)])
    oi.main(["--out", str(staged), "--stage", "merge"])
    for f in ("s1_3p__T_cell.npz", "s1_5p__B_cell.npz"):
        a, b = np.load(one / f), np.load(staged / f)
        assert list(a["conditions"]) == list(b["conditions"])
        assert np.array_equal(a["token_ids"], b["token_ids"])
        for k in ("n_heldout", "target_sum", "pred_rank", "p_correct", "cls", "cls_cells"):
            assert np.allclose(a[k], b[k], atol=1e-4), k
        assert a["cls"].shape[1] == 12 and np.abs(a["cls"][0]).sum() > 0
