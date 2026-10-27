from __future__ import annotations

# --- release paths: repo root on sys.path, results written to results/synthetic ---
import sys as _sys
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO))
_RESULTS = _REPO / "results" / "synthetic"
_RESULTS.mkdir(parents=True, exist_ok=True)

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from dcb_nmf.baselines import (
    separate_auxiva_all,
    separate_ds_all,
    separate_gev_all,
    separate_mpdr_all,
    separate_mvdr_mask_all,
    separate_mwf_all,
    separate_ratio_all,
    separate_zf_all,
)
from dcb_nmf.method import (
    DCBConfig,
    dcb_nmf_separate,
    separate_lcmv_all,
    separate_mvdr_all,
    separate_nmf_all,
)
from dcb_nmf.metrics import permute_si_sdri
from dcb_nmf.mix import (
    cocktail_talker_with_envelope,
    controlled_overlap_envelopes,
    doas_for_separation,
    make_linear_array,
    simulate_cocktail,
)

SR = 16000
DURATION = 3.0
N_FFT = 1024
HOP = 256
SNR_DB = 10.0
N_TRIALS = 5

ANGLES = [90.0, 120.0, 150.0]
OVERLAPS = [0.25, 0.50, 0.75]
ALPHA_GRID = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]

FORMANT1 = [(500, 80, 1.0), (1100, 100, 0.7), (2400, 150, 0.45)]
FORMANT2 = [(350, 70, 1.0), (1900, 120, 0.85), (2800, 160, 0.35)]

BASELINE_FNS = {
    "NMF": separate_nmf_all,
    "DS": separate_ds_all,
    "Ratio+DS": separate_ratio_all,
    "MPDR": separate_mpdr_all,
    "MVDR": separate_mvdr_all,
    "LCMV": separate_lcmv_all,
    "GEV": separate_gev_all,
    "MWF": separate_mwf_all,
    "ZF": separate_zf_all,
    "AuxIVA": separate_auxiva_all,
    "MVDR+Mask": separate_mvdr_mask_all,
}


def make_scene(sep_deg, overlap, seed):
    rng = np.random.default_rng(seed)
    n = int(DURATION * SR)
    env1, env2, realized = controlled_overlap_envelopes(n, SR, overlap, rng)
    s1 = cocktail_talker_with_envelope(DURATION, SR, 110.0, FORMANT1, env1, rng)
    s2 = cocktail_talker_with_envelope(DURATION, SR, 200.0, FORMANT2, env2, rng)
    doas = doas_for_separation(sep_deg)
    mic = make_linear_array(n_mics=4, spacing=0.05)
    mix, images, _ = simulate_cocktail(
        [s1, s2], doas, sr=SR, mic_pos=mic, snr_db=SNR_DB, rng=rng
    )
    return mix, images[:, :, 0], doas, realized


def score(est, refs, mix_ref):
    _s, _p, mean = permute_si_sdri(est, refs, mix_ref)
    return float(mean)


def main():
    mic = make_linear_array(n_mics=4, spacing=0.05)
    cfg = DCBConfig(
        n_fft=N_FFT, hop=HOP, n_sources=2, n_outer=2,
        n_nmf_iter=60, n_fuse_iter=40, seed=0,
    )
    records = []
    t0 = time.time()
    for angle in ANGLES:
        for overlap in OVERLAPS:
            for trial in range(N_TRIALS):
                seed = 1000 + int(angle) * 17 + int(overlap * 100) * 3 + trial
                mix, refs, doas, realized = make_scene(angle, overlap, seed)
                mref = mix[:, 0]
                row = {
                    "angle": angle, "overlap": overlap, "trial": trial,
                    "realized": realized, "scores": {},
                }
                for name, fn in BASELINE_FNS.items():
                    row["scores"][name] = score(fn(mix, SR, mic, cfg, doas), refs, mref)
                for a in ALPHA_GRID:
                    est, _ = dcb_nmf_separate(
                        mix, SR, mic_pos=mic, cfg=replace(cfg, alpha=float(a)), doas=doas
                    )
                    row["scores"][f"DCB-NMF@a{a:.1f}"] = score(est, refs, mref)
                records.append(row)
                print(
                    f"ang={int(angle)} ov={overlap:.2f} trial={trial} "
                    f"realized={realized:.2f} "
                    f"dcb0.5={row['scores']['DCB-NMF@a0.5']:.2f} "
                    f"mvdrmask={row['scores']['MVDR+Mask']:.2f} "
                    f"[{time.time()-t0:.0f}s]",
                    flush=True,
                )
    out = _RESULTS / "paper_raw.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(records, indent=1), encoding="utf-8")
    print("wrote", out, f"({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
