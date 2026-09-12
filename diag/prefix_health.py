#!/usr/bin/env python
"""Is the fusion stack's prefix collapsed? 30-second check against a middle.pt.

    python diag/prefix_health.py outputs/checkpoints/stage1/best

The number that matters is ||dev||/||mean||: the object-specific share of the prefix, measured
over real cached embeddings. Reference points, all measured on this repo:

    shipped v6 (broken)                    0.094
    fresh Q-Former, discriminative loss    0.83 - 1.28
    anything below ~0.3                    still collapsing

Also prints the prefix vector norm, which must land near the LLM's own token-embedding norm
(~0.89 for Qwen3.5-9B). v6 shipped at 30.6, i.e. 34x out of distribution.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from captioner.data.dataset import ModalityCacheReader  # noqa: E402
from captioner.model.captioner import FusionStack  # noqa: E402

MAXT = {"image": 576, "spectra": 512, "lightcurve": 243}
DIM = {"image": 768, "spectra": 768, "lightcurve": 384}


def spec_hash_for(cache_root: Path, modality: str) -> str | None:
    d = cache_root / modality
    subs = [p.name for p in d.iterdir() if p.is_dir()] if d.is_dir() else []
    return subs[0] if subs else None


def main() -> int:
    ckpt = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/checkpoints/stage1/best")
    cache_root = Path(sys.argv[2] if len(sys.argv) > 2 else "outputs/cache")
    state = torch.load(ckpt / "middle.pt", map_location="cpu", weights_only=False)
    d_llm = state["adapter.net.3.weight"].shape[0]
    d_shared = state["qformer.query_embed"].shape[1]
    n_queries = state["qformer.query_embed"].shape[0]
    n_layers = len({k.split(".")[2] for k in state if k.startswith("qformer.layers.")})
    mods = sorted({k.split(".")[1] for k in state if k.startswith("projectors.")})

    fs = FusionStack(
        {m: DIM[m] for m in mods}, d_shared, d_llm,
        dict(n_queries=n_queries, d_model=d_shared, n_layers=n_layers, n_heads=6, ffn_mult=4, dropout=0.1),
        2, 0.1,
    )
    if "adapter.out_norm.weight" not in state:
        # Pre-fix checkpoint (v5/v6): the adapter had no output norm or scale. Run it the way it
        # was actually trained so the "before" number is honest, rather than refusing to load.
        print("  [legacy checkpoint: adapter has no output norm — measuring as-trained]")
        fs.adapter.forward = (lambda net: (lambda x: net(x)))(fs.adapter.net)
        fs.load_state_dict(state, strict=False)
    else:
        fs.load_state_dict(state)
    fs.eval()
    print(f"{ckpt}  d_shared={d_shared} n_queries={n_queries} n_layers={n_layers} d_llm={d_llm}\n")
    print(f"  {'modality':<12}{'||dev||/||mean||':>18}{'prefix norm':>14}{'pairwise cos':>16}   verdict")
    worst = 1e9
    worst_cos = 0.0
    for mod in mods:
        h = spec_hash_for(cache_root, mod)
        if h is None:
            print(f"  {mod:<12}{'(no cache)':>18}")
            continue
        rd = ModalityCacheReader(cache_root, mod, h)
        ids = list(rd.index.index)[:64]
        X = torch.zeros(len(ids), MAXT[mod], DIM[mod])
        M = torch.ones(len(ids), MAXT[mod], dtype=torch.bool)
        for i, o in enumerate(ids):
            a = torch.from_numpy(rd.get(o))
            k = min(a.shape[0], MAXT[mod])
            X[i, :k] = a[:k]
            M[i, :k] = False
        batch = {m: {"tokens": torch.zeros(len(ids), MAXT[m], DIM[m]),
                     "mask": torch.ones(len(ids), MAXT[m], dtype=torch.bool)} for m in mods}
        batch[mod] = {"tokens": X, "mask": M}
        with torch.no_grad():
            P = fs(batch)
        F = P.reshape(len(ids), -1)
        ratio = ((F - F.mean(0)).norm(dim=1).mean() / F.mean(0).norm()).item()
        norm = P.reshape(-1, d_llm).norm(dim=1).mean().item()
        # Mean pairwise cosine between DIFFERENT objects' prefixes — the same quantity the
        # original bug report measured by hand (quasar vs galaxy vs zeros vs noise, all >0.96).
        # Unlike `ratio` this needs no reference scale, so it stays comparable across adapter
        # geometries: the output LayerNorm added in the fix constrains every prefix vector to the
        # same magnitude, which mechanically lowers `ratio` even when conditioning improves.
        Fn = F / F.norm(dim=1, keepdim=True)
        S = (Fn @ Fn.T)
        off = S[~torch.eye(len(F), dtype=torch.bool)]
        pair_cos = off.mean().item()
        worst = min(worst, ratio)
        worst_cos = max(worst_cos, pair_cos)
        verdict = "COLLAPSED" if pair_cos > 0.95 else ("weak" if pair_cos > 0.85 else "ok")
        print(f"  {mod:<12}{ratio:>18.4f}{norm:>14.3f}{pair_cos:>16.4f}   {verdict}")
    # Spectra split by survey: DESI and SDSS go through different AION codecs, and DESI was
    # 331/334 in test before v7. A single spectra number averages the two and hides that.
    if "spectra" in mods:
        h = spec_hash_for(cache_root, "spectra")
        man = Path("outputs/manifest/manifest.parquet")
        if h is not None and man.exists():
            import pandas as pd

            survey_of = dict(
                zip(*pd.read_parquet(man, columns=["object_id", "survey"]).to_dict("list").values())
            )
            rd = ModalityCacheReader(cache_root, "spectra", h)
            print()
            for survey in ("desi", "sdss"):
                sel = [o for o in rd.index.index if survey_of.get(o) == survey][:48]
                if len(sel) < 8:
                    print(f"  {'spectra/' + survey:<12}{'(only ' + str(len(sel)) + ' cached)':>18}")
                    continue
                X = torch.zeros(len(sel), MAXT["spectra"], DIM["spectra"])
                M = torch.ones(len(sel), MAXT["spectra"], dtype=torch.bool)
                for i, o in enumerate(sel):
                    a = torch.from_numpy(rd.get(o))
                    k = min(a.shape[0], MAXT["spectra"])
                    X[i, :k] = a[:k]
                    M[i, :k] = False
                batch = {m: {"tokens": torch.zeros(len(sel), MAXT[m], DIM[m]),
                             "mask": torch.ones(len(sel), MAXT[m], dtype=torch.bool)} for m in mods}
                batch["spectra"] = {"tokens": X, "mask": M}
                with torch.no_grad():
                    P = fs(batch)
                F = P.reshape(len(sel), -1)
                ratio = ((F - F.mean(0)).norm(dim=1).mean() / F.mean(0).norm()).item()
                Fn = F / F.norm(dim=1, keepdim=True)
                S = Fn @ Fn.T
                pc = S[~torch.eye(len(F), dtype=torch.bool)].mean().item()
                print(f"  {'spectra/' + survey:<12}{ratio:>18.4f}{'':>14}{pc:>16.4f}   n={len(sel)}")

    print()
    print("  reference — published v6: pairwise cos 0.988-0.999, prefix norm 30.6 (Qwen's is 0.89)")
    if worst_cos > 0.95:
        print("FAIL: different objects still map to near-identical prefixes. See DIAGNOSIS.md.")
        return 1
    print("PASS: different objects map to distinguishable prefixes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
