from __future__ import annotations

# --- release paths: repo root on sys.path, results written to results/synthetic ---
import sys as _sys
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO))
_RESULTS = _REPO / "results" / "synthetic"
_RESULTS.mkdir(parents=True, exist_ok=True)

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from ablate import dcb_variant
from dcb_nmf.method import DCBConfig
from dcb_nmf.mix import make_linear_array
from eval_paper import ANGLES, N_TRIALS, OVERLAPS, SR, make_scene, score

ROOT = Path(__file__).resolve().parent

BASE = dict(mean="arith", lcmv_in_consensus=False)  # dev-selected
ALPHA_STAR = 0.4
ALPHAS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

ABLATIONS = {
    "no_lcmv": dict(BASE, use_lcmv=False),
    "lcmv_in_consensus": dict(BASE, lcmv_in_consensus=True),
    "geo_mean": dict(BASE, mean="geo"),
    "no_shared_basis": dict(BASE, shared_basis=False),
    "open_loop": dict(BASE, closed_loop=False),
}


def main():
    mic = make_linear_array(n_mics=4, spacing=0.05)
    cfg0 = DCBConfig(
        n_fft=1024, hop=256, n_sources=2, n_outer=2,
        n_nmf_iter=60, n_fuse_iter=40, seed=0,
    )
    records = []
    t0 = time.time()
    for angle in ANGLES:
        for overlap in OVERLAPS:
            for trial in range(N_TRIALS):
                seed = 1000 + int(angle) * 17 + int(overlap * 100) * 3 + trial
                mix, refs, doas, _r = make_scene(angle, overlap, seed)
                mref = mix[:, 0]
                row = {"angle": angle, "overlap": overlap, "trial": trial, "scores": {}}
                for a in ALPHAS:
                    est = dcb_variant(
                        mix, SR, mic, replace(cfg0, alpha=float(a)), doas, **BASE
                    )
                    row["scores"][f"DCB@a{a:.1f}"] = score(est, refs, mref)
                row["scores"]["full"] = row["scores"][f"DCB@a{ALPHA_STAR:.1f}"]
                for name, kw in ABLATIONS.items():
                    est = dcb_variant(
                        mix, SR, mic, replace(cfg0, alpha=ALPHA_STAR), doas, **kw
                    )
                    row["scores"][name] = score(est, refs, mref)
                records.append(row)
                print(
                    f"ang={int(angle)} ov={overlap:.2f} t={trial} "
                    f"full={row['scores']['full']:.1f} "
                    + " ".join(f"{k}={row['scores'][k]:.1f}" for k in ABLATIONS)
                    + f" [{time.time()-t0:.0f}s]",
                    flush=True,
                )
    (_RESULTS / "test_final.json").write_text(
        json.dumps(records, indent=1), encoding="utf-8"
    )
    print("\n=== TEST summary ===")
    for a in ALPHAS:
        v = np.array([r["scores"][f"DCB@a{a:.1f}"] for r in records])
        print(f"  alpha={a:.1f}  {v.mean():6.2f} +/- {v.std():4.2f}")
    base = np.array([r["scores"]["full"] for r in records]).mean()
    for name in ABLATIONS:
        v = np.array([r["scores"][name] for r in records])
        print(f"  {name:20s} {v.mean():6.2f}  delta={v.mean()-base:+.2f}")


if __name__ == "__main__":
    main()
