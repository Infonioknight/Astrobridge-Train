#!/usr/bin/env python
"""Do the cached embeddings still tell objects apart? Runs straight after 02_cache_embeddings.

Reports, per modality, the pairwise cosine between objects AFTER removing the shared mean.

Read this as a smoke test, not a proof. It catches TOTAL collapse (every object encoded to the
same vector). It would NOT have caught the v5/v6 spectra bug on its own: that cache scored
||dev||/||mean|| = 0.386 and looked healthy here, because the unsorted wavelength grid still
produced per-object variation — just variation that carried no physical information (a linear
class probe on it scored below its own majority baseline).

The signal that DOES show it is `centred-cos mean`: broken spectra sat at +0.244 while the two
healthy modalities sat near zero (image -0.015, lightcurve +0.022). Every object being
positively correlated with every other object after centring means they share one dominant
spurious direction. Flagged as SUSPECT above 0.15.

The real guard for that bug is the strictly-increasing-wavelength assertion in
encoders/aion_spectrum.py, which hard-errors rather than caching garbage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def main() -> int:
    cache_root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/cache")
    bad = False
    for mod_dir in sorted(p for p in cache_root.iterdir() if p.is_dir()):
        for spec_dir in sorted(p for p in mod_dir.iterdir() if p.is_dir()):
            shards = sorted(spec_dir.glob("shard_*.npy"))
            if not shards:
                continue
            X = np.load(shards[0]).astype(np.float32)
            n = min(len(X), 64)
            F = X[:n].reshape(n, -1)
            Fc = F - F.mean(0, keepdims=True)
            Fn = Fc / (np.linalg.norm(Fc, axis=1, keepdims=True) + 1e-9)
            S = Fn @ Fn.T
            off = S[~np.eye(n, dtype=bool)]
            spread = float(off.std())
            ratio = float(np.linalg.norm(Fc, axis=1).mean() / (np.linalg.norm(F.mean(0)) + 1e-9))
            if ratio < 0.05:
                flag, bad = "COLLAPSED", True
            elif abs(off.mean()) > 0.15:
                flag, bad = "SUSPECT", True
            else:
                flag = "ok"
            print(
                f"  {mod_dir.name:<12} {spec_dir.name[:10]:<12} n={n:<4} "
                f"centred-cos mean={off.mean():+.4f} sd={spread:.4f}  "
                f"||dev||/||mean||={ratio:.4f}  {flag}"
            )
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
