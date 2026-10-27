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

from dcb_nmf.method import DCBConfig, _beamform_source, _identify_target, _nmf_init
from dcb_nmf.metrics import permute_si_sdri
from dcb_nmf.nmf import kl_nmf
from dcb_nmf.stft import analysis, synthesis
from eval_paper import ANGLES, N_TRIALS, OVERLAPS, SR, make_scene, score


def fuse(views, n_bases, n_iter, rng, consensus_idx, mean="geo"):
    views = [np.maximum(np.asarray(m, dtype=np.float64), 0.0) for m in views]
    n_freq = views[0].shape[0]
    w, h = kl_nmf(np.vstack(views), n_bases=n_bases, n_iter=n_iter, rng=rng)
    recons = [
        np.maximum(w[i * n_freq : (i + 1) * n_freq] @ h, 1e-10)
        for i in range(len(views))
    ]
    chosen = np.stack([recons[i] for i in consensus_idx], axis=0)
    if mean == "geo":
        return np.exp(np.mean(np.log(chosen), axis=0))
    return np.mean(chosen, axis=0)


def dcb_variant(
    mix, sr, mic_pos, cfg, doas,
    use_lcmv=True,          # form the LCMV branch at all
    lcmv_in_consensus=False,  # include LCMV in the geometric-mean readout
    closed_loop=True,       # feed fused magnitudes back into masks/SCMs
    shared_basis=True,      # joint factorization vs per-view independent
    mean="geo",             # geometric vs arithmetic consensus
):
    rng = np.random.default_rng(cfg.seed)
    n_samples = mix.shape[0]
    X = analysis(mix, sr=sr, n_fft=cfg.n_fft, hop=cfg.hop)
    masks, recon = _nmf_init(np.abs(X[:, :, cfg.ref_mic]), cfg, rng)
    n_src = cfg.n_sources
    if doas is not None and mic_pos is not None:
        order, remaining = [], list(range(n_src))
        for doa in doas[:n_src]:
            pick = _identify_target(
                X, [masks[i] for i in remaining], sr, cfg.n_fft, doa, mic_pos
            )
            order.append(remaining.pop(pick))
        order.extend(remaining)
        masks = [masks[i] for i in order]
        recon = [recon[i] for i in order]

    priors = [np.maximum(r, 1e-10) for r in recon]
    fused_specs = [None] * n_src
    n_outer = cfg.n_outer if closed_loop else 1

    for _ in range(n_outer):
        mags = []
        for k in range(n_src):
            others = [masks[j] for j in range(n_src) if j != k]
            hint = doas[k] if doas is not None and k < len(doas) else None
            y_mvdr, y_lcmv, ok = _beamform_source(
                X, masks[k], others, cfg.loading, cfg.lcmv_align_max,
                sr, cfg.n_fft, mic_pos, hint,
            )
            masked_mvdr = masks[k] * np.abs(y_mvdr)
            views = [np.abs(y_mvdr), priors[k], masked_mvdr]
            consensus = [0, 1, 2]
            if ok and use_lcmv:
                views.append(np.abs(y_lcmv))
                if lcmv_in_consensus:
                    consensus.append(3)
            if shared_basis:
                mag = fuse(
                    views, cfg.n_fuse_bases, cfg.n_fuse_iter, rng,
                    tuple(consensus), mean=mean,
                )
            else:  # factorize each view independently, then combine
                recons = []
                for v in views:
                    w, h = kl_nmf(
                        np.maximum(v, 0.0), n_bases=cfg.n_fuse_bases,
                        n_iter=cfg.n_fuse_iter, rng=rng,
                    )
                    recons.append(np.maximum(w @ h, 1e-10))
                chosen = np.stack([recons[i] for i in consensus], axis=0)
                mag = (
                    np.exp(np.mean(np.log(chosen), axis=0))
                    if mean == "geo" else np.mean(chosen, axis=0)
                )
            a = float(np.clip(cfg.alpha, 0.0, 1.0))
            mag = a * mag + (1.0 - a) * masked_mvdr
            fused_specs[k] = mag * np.exp(1j * np.angle(y_mvdr))
            mags.append(mag)
        total = sum(mags) + 1e-10
        masks = [m / total for m in mags]
        priors = mags

    return np.stack(
        [
            synthesis(s, sr=sr, n_fft=cfg.n_fft, hop=cfg.hop, length=n_samples)
            for s in fused_specs
        ],
        axis=0,
    )


VARIANTS = {
    "full": {},
    "no_lcmv": {"use_lcmv": False},
    "lcmv_in_consensus": {"lcmv_in_consensus": True},
    "open_loop": {"closed_loop": False},
    "no_shared_basis": {"shared_basis": False},
    "arith_mean": {"mean": "arith"},
}


def main():
    from dcb_nmf.mix import make_linear_array

    mic = make_linear_array(n_mics=4, spacing=0.05)
    cfg = DCBConfig(
        n_fft=1024, hop=256, n_sources=2, n_outer=2,
        n_nmf_iter=60, n_fuse_iter=40, seed=0, alpha=0.5,
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
                for name, kw in VARIANTS.items():
                    est = dcb_variant(mix, SR, mic, cfg, doas, **kw)
                    row["scores"][name] = score(est, refs, mref)
                records.append(row)
                print(
                    f"ang={int(angle)} ov={overlap:.2f} t={trial} "
                    + " ".join(f"{k}={row['scores'][k]:.1f}" for k in VARIANTS)
                    + f" [{time.time()-t0:.0f}s]",
                    flush=True,
                )
    out = _RESULTS / "ablation_raw.json"
    out.write_text(json.dumps(records, indent=1), encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
