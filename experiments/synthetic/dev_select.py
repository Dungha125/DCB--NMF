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
from eval_paper import ANGLES, OVERLAPS, SR, make_scene, score

ROOT = Path(__file__).resolve().parent
DEV_TRIALS = 3
DEV_SEED_OFFSET = 500000  # disjoint from the test seeds (1000 + ...)

CONFIGS = {
    "geo_lcmvOut": dict(mean="geo", lcmv_in_consensus=False),
    "geo_lcmvIn": dict(mean="geo", lcmv_in_consensus=True),
    "arith_lcmvOut": dict(mean="arith", lcmv_in_consensus=False),
    "arith_lcmvIn": dict(mean="arith", lcmv_in_consensus=True),
}
ALPHAS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]


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
            for trial in range(DEV_TRIALS):
                seed = DEV_SEED_OFFSET + int(angle) * 17 + int(overlap * 100) * 3 + trial
                mix, refs, doas, _r = make_scene(angle, overlap, seed)
                mref = mix[:, 0]
                row = {"angle": angle, "overlap": overlap, "trial": trial, "scores": {}}
                # stage 1: pick consensus rule at alpha = 0.5
                for name, kw in CONFIGS.items():
                    est = dcb_variant(
                        mix, SR, mic, replace(cfg0, alpha=0.5), doas, **kw
                    )
                    row["scores"][name] = score(est, refs, mref)
                # stage 2: alpha sweep for every consensus rule
                for name, kw in CONFIGS.items():
                    for a in ALPHAS:
                        est = dcb_variant(
                            mix, SR, mic, replace(cfg0, alpha=float(a)), doas, **kw
                        )
                        row["scores"][f"{name}@a{a:.1f}"] = score(est, refs, mref)
                records.append(row)
                print(
                    f"ang={int(angle)} ov={overlap:.2f} t={trial} "
                    + " ".join(f"{k}={row['scores'][k]:.1f}" for k in CONFIGS)
                    + f" [{time.time()-t0:.0f}s]",
                    flush=True,
                )
    out = _RESULTS / "dev_raw.json"
    out.write_text(json.dumps(records, indent=1), encoding="utf-8")

    print("\n=== DEV summary (mean SI-SDRi, dB) ===")
    for name in CONFIGS:
        v = np.array([r["scores"][name] for r in records])
        print(f"  alpha=0.5  {name:16s} {v.mean():6.2f}")
    print()
    best = None
    for name in CONFIGS:
        for a in ALPHAS:
            v = np.array([r["scores"][f"{name}@a{a:.1f}"] for r in records])
            print(f"  {name:16s} a={a:.1f}  {v.mean():6.2f}")
            if best is None or v.mean() > best[2]:
                best = (name, a, float(v.mean()))
    print(f"\nDEV winner: {best[0]}  alpha={best[1]}  ({best[2]:.2f} dB)")
    (_RESULTS / "dev_choice.json").write_text(
        json.dumps({"config": best[0], "alpha": best[1], "dev_score": best[2]}, indent=1)
    )


if __name__ == "__main__":
    main()
