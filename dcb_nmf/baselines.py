from __future__ import annotations

import numpy as np

from .beamformers import (
    apply_beamformer,
    ds_weights,
    estimate_scm,
    gev_weights,
    mpdr_weights,
    mwf_weights,
    mvdr_weights,
    zf_weights,
)
from .method import (
    DCBConfig,
    _init_masks,
    _source_steering,
    _steering_alignment,
    dcb_nmf_separate,
    separate_lcmv_all,
    separate_mvdr_all,
    separate_nmf_all,
)
from .stft import analysis, synthesis


def _source_stats(
    X: np.ndarray,
    masks: list[np.ndarray],
    k: int,
    sr: int,
    cfg: DCBConfig,
    mic_pos: np.ndarray | None,
    doas: list[float] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Return Φ_s, Φ_n, a_s, list of interferer steerings for source k."""
    others = [masks[j] for j in range(len(masks)) if j != k]
    mask_n = np.zeros_like(masks[k])
    for m in others:
        mask_n = mask_n + m
    mask_n = np.clip(mask_n, 0.0, 1.0)
    if float(mask_n.mean()) < 1e-6:
        mask_n = np.clip(1.0 - masks[k], 0.0, 1.0)
    phi_s = estimate_scm(X, masks[k])
    phi_n = estimate_scm(X, mask_n)
    hint = doas[k] if doas is not None and k < len(doas) else None
    a_s = _source_steering(
        phi_s,
        sr,
        cfg.n_fft,
        mic_pos,
        doa_hint=hint,
        mode=cfg.steering_mode,
        phi_n=phi_n,
        loading=cfg.loading,
    )
    interfs = []
    for m in others:
        if float(m.mean()) < 1e-5:
            continue
        phi_i = estimate_scm(X, m)
        a_i = _source_steering(
            phi_i,
            sr,
            cfg.n_fft,
            mic_pos,
            doa_hint=None,
            mode=cfg.steering_mode,
            phi_n=phi_n,
            loading=cfg.loading,
        )
        if _steering_alignment(a_s, a_i) < cfg.lcmv_align_max:
            interfs.append(a_i)
    return phi_s, phi_n, a_s, interfs


def _synth_all(specs: list[np.ndarray], sr: int, cfg: DCBConfig, n: int) -> np.ndarray:
    return np.stack(
        [synthesis(y, sr=sr, n_fft=cfg.n_fft, hop=cfg.hop, length=n) for y in specs],
        axis=0,
    )


def separate_ds_all(
    mix: np.ndarray,
    sr: int,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
    doas: list[float] | None = None,
) -> np.ndarray:
    """Delay-and-sum beamforming steered by NMF-mask DOA."""
    cfg = cfg or DCBConfig()
    X, masks = _init_masks(mix, sr, cfg, mic_pos, doas)
    outs = []
    for k in range(len(masks)):
        _ps, _pn, a_s, _ = _source_stats(X, masks, k, sr, cfg, mic_pos, doas)
        outs.append(apply_beamformer(X, ds_weights(a_s)))
    return _synth_all(outs, sr, cfg, mix.shape[0])


def separate_mpdr_all(
    mix: np.ndarray,
    sr: int,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
    doas: list[float] | None = None,
) -> np.ndarray:
    """MPDR / Capon using mixture SCM + NMF steering."""
    cfg = cfg or DCBConfig()
    X, masks = _init_masks(mix, sr, cfg, mic_pos, doas)
    ones = np.ones(X.shape[:2], dtype=np.float64)
    phi_xx = estimate_scm(X, ones)
    outs = []
    for k in range(len(masks)):
        _ps, _pn, a_s, _ = _source_stats(X, masks, k, sr, cfg, mic_pos, doas)
        outs.append(apply_beamformer(X, mpdr_weights(phi_xx, a_s, cfg.loading)))
    return _synth_all(outs, sr, cfg, mix.shape[0])


def separate_gev_all(
    mix: np.ndarray,
    sr: int,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
    doas: list[float] | None = None,
) -> np.ndarray:
    """GEV (max-SNR) beamformer from NMF speech/noise SCMs."""
    cfg = cfg or DCBConfig()
    X, masks = _init_masks(mix, sr, cfg, mic_pos, doas)
    outs = []
    for k in range(len(masks)):
        phi_s, phi_n, _a, _ = _source_stats(X, masks, k, sr, cfg, mic_pos, doas)
        outs.append(apply_beamformer(X, gev_weights(phi_s, phi_n, cfg.loading)))
    return _synth_all(outs, sr, cfg, mix.shape[0])


def separate_mwf_all(
    mix: np.ndarray,
    sr: int,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
    doas: list[float] | None = None,
) -> np.ndarray:
    """Multichannel Wiener filter from NMF SCMs."""
    cfg = cfg or DCBConfig()
    X, masks = _init_masks(mix, sr, cfg, mic_pos, doas)
    outs = []
    for k in range(len(masks)):
        phi_s, phi_n, _a, _ = _source_stats(X, masks, k, sr, cfg, mic_pos, doas)
        outs.append(
            apply_beamformer(
                X, mwf_weights(phi_s, phi_n, ref_mic=cfg.ref_mic, loading=cfg.loading)
            )
        )
    return _synth_all(outs, sr, cfg, mix.shape[0])


def separate_zf_all(
    mix: np.ndarray,
    sr: int,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
    doas: list[float] | None = None,
) -> np.ndarray:
    """Zero-forcing spatial nulling of interferers."""
    cfg = cfg or DCBConfig()
    X, masks = _init_masks(mix, sr, cfg, mic_pos, doas)
    outs = []
    for k in range(len(masks)):
        _ps, _pn, a_s, interfs = _source_stats(X, masks, k, sr, cfg, mic_pos, doas)
        n_mics = X.shape[-1]
        interfs = interfs[: max(0, n_mics - 1)]
        outs.append(apply_beamformer(X, zf_weights(a_s, interfs)))
    return _synth_all(outs, sr, cfg, mix.shape[0])


def _auxiva_freq(
    Xf: np.ndarray,
    n_iter: int = 30,
    eps: float = 1e-8,
) -> np.ndarray:
    """AuxIVA iterative projection on one frequency. Xf: (T, M) complex."""
    t_frames, n_mics = Xf.shape
    w = np.eye(n_mics, dtype=np.complex128)
    for _ in range(n_iter):
        y = Xf @ np.conj(w.T)
        r = np.sqrt(np.maximum(np.abs(y) ** 2, eps))
        for n in range(n_mics):
            g = 1.0 / r[:, n]
            v = (Xf.T * g) @ np.conj(Xf) / t_frames
            v = 0.5 * (v + np.conj(v.T)) + eps * np.eye(n_mics)
            w_n = np.linalg.solve(w @ v, np.eye(n_mics)[:, n])
            denom = np.sqrt(np.maximum(np.real(np.conj(w_n) @ v @ w_n), eps))
            w[n] = w_n / denom
    return Xf @ np.conj(w.T)


def _pca_project(X: np.ndarray, n_src: int) -> np.ndarray:
    """Project (F,T,M) mix onto top-n_src PCA subspace -> (F,T,n_src)."""
    n_freq, n_frames, _ = X.shape
    ones = np.ones((n_freq, n_frames), dtype=np.float64)
    phi = estimate_scm(X, ones)
    xp = np.zeros((n_freq, n_frames, n_src), dtype=np.complex128)
    for f in range(n_freq):
        _evals, evecs = np.linalg.eigh(phi[f])
        u = evecs[:, -n_src:]
        xp[f] = X[f] @ np.conj(u)
    return xp


def _match_channels_to_masks(
    Y: np.ndarray, masks: list[np.ndarray]
) -> list[int]:
    """Greedy TF-magnitude correlation assignment; returns mask->channel index."""
    n_src = len(masks)
    scores = np.zeros((n_src, n_src))
    for s in range(n_src):
        for c in range(n_src):
            a = masks[s].ravel()
            b = np.abs(Y[:, :, c]).ravel()
            if a.std() < 1e-12 or b.std() < 1e-12:
                scores[s, c] = -1.0
            else:
                scores[s, c] = float(np.corrcoef(a, b)[0, 1])
    assign = [-1] * n_src
    score_work = np.nan_to_num(scores, nan=-1.0)
    for _ in range(n_src):
        s, c = np.unravel_index(np.argmax(score_work), score_work.shape)
        assign[int(s)] = int(c)
        score_work[int(s), :] = -np.inf
        score_work[:, int(c)] = -np.inf
    return assign


def _ilrma_demix(
    Xp: np.ndarray,
    n_bases: int = 10,
    n_iter: int = 40,
    eps: float = 1e-8,
    seed: int = 0,
) -> np.ndarray:
    """Determined ILRMA (IS-NMF source model + IP demixing).

    Xp : (F, T, N) PCA-projected mix. Returns separated Y of same shape.
    Demixing uses the AuxIVA convention Y = X W^H with rows of W as filters.
    """
    n_freq, n_frames, n_src = Xp.shape
    rng = np.random.default_rng(seed)
    w = np.broadcast_to(
        np.eye(n_src, dtype=np.complex128), (n_freq, n_src, n_src)
    ).copy()
    t_mat = [rng.random((n_freq, n_bases)) + 0.1 for _ in range(n_src)]
    v_mat = [rng.random((n_bases, n_frames)) + 0.1 for _ in range(n_src)]

    for _ in range(n_iter):
        # Y[f,t,n] = sum_m X[f,t,m] conj(W[f,n,m])
        y = np.einsum("ftm,fnm->ftn", Xp, np.conj(w))
        p = np.maximum(np.abs(y) ** 2, eps)
        for n in range(n_src):
            pn = p[:, :, n]
            tn, vn = t_mat[n], v_mat[n]
            lam_n = np.maximum(tn @ vn, eps)
            tn *= ((pn / (lam_n**2)) @ vn.T) / (
                np.maximum(1.0 / lam_n, eps) @ vn.T + eps
            )
            tn = np.maximum(tn, eps)
            lam_n = np.maximum(tn @ vn, eps)
            vn *= (tn.T @ (pn / (lam_n**2))) / (
                tn.T @ np.maximum(1.0 / lam_n, eps) + eps
            )
            vn = np.maximum(vn, eps)
            # scale fix: absorb column norms of T into V
            scale = np.maximum(tn.sum(axis=0, keepdims=True), eps)
            tn = tn / scale
            vn = vn * scale.T
            t_mat[n], v_mat[n] = tn, vn
        lam = np.stack(
            [np.maximum(t_mat[n] @ v_mat[n], eps) for n in range(n_src)], axis=-1
        )
        for n in range(n_src):
            inv_lam = 1.0 / np.maximum(lam[:, :, n], eps)
            for f in range(n_freq):
                xf = Xp[f]
                g = inv_lam[f]
                v = (xf.T * g) @ np.conj(xf) / n_frames
                v = 0.5 * (v + np.conj(v.T)) + eps * np.eye(n_src)
                try:
                    w_n = np.linalg.solve(w[f] @ v, np.eye(n_src)[:, n])
                except np.linalg.LinAlgError:
                    continue
                denom = np.sqrt(np.maximum(np.real(np.conj(w_n) @ v @ w_n), eps))
                w[f, n] = w_n / denom
    return np.einsum("ftm,fnm->ftn", Xp, np.conj(w))


def separate_ilrma_all(
    mix: np.ndarray,
    sr: int,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
    doas: list[float] | None = None,
) -> np.ndarray:
    """PCA + determined ILRMA; permute outputs to NMF masks."""
    cfg = cfg or DCBConfig()
    n_src = cfg.n_sources
    X, masks = _init_masks(mix, sr, cfg, mic_pos, doas)
    xp = _pca_project(X, n_src)
    y = _ilrma_demix(
        xp,
        n_bases=min(cfg.n_fuse_bases, 10),
        n_iter=40,
        seed=cfg.seed,
    )
    assign = _match_channels_to_masks(y, masks)
    # Keep demixed waveforms; NMF masks only resolve permutation.
    specs = [y[:, :, assign[s]] for s in range(n_src)]
    return _synth_all(specs, sr, cfg, mix.shape[0])


def separate_mvdr_mask_all(
    mix: np.ndarray,
    sr: int,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
    doas: list[float] | None = None,
) -> np.ndarray:
    """MVDR followed by NMF soft-mask postfilter."""
    cfg = cfg or DCBConfig()
    X, masks = _init_masks(mix, sr, cfg, mic_pos, doas)
    outs = []
    for k in range(len(masks)):
        phi_s, phi_n, a_s, _ = _source_stats(X, masks, k, sr, cfg, mic_pos, doas)
        y = apply_beamformer(X, mvdr_weights(phi_n, a_s, cfg.loading))
        outs.append(masks[k] * y)
    return _synth_all(outs, sr, cfg, mix.shape[0])


def separate_auxiva_all(
    mix: np.ndarray,
    sr: int,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
    doas: list[float] | None = None,
) -> np.ndarray:
    """PCA + AuxIVA; permute outputs to NMF masks (or DOA order)."""
    cfg = cfg or DCBConfig()
    n_src = cfg.n_sources
    X, masks = _init_masks(mix, sr, cfg, mic_pos, doas)
    xp = _pca_project(X, n_src)
    y = np.zeros_like(xp)
    for f in range(xp.shape[0]):
        y[f] = _auxiva_freq(xp[f], n_iter=30)
    assign = _match_channels_to_masks(y, masks)
    specs = [y[:, :, assign[s]] for s in range(n_src)]
    return _synth_all(specs, sr, cfg, mix.shape[0])


def separate_ratio_all(
    mix: np.ndarray,
    sr: int,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
    doas: list[float] | None = None,
) -> np.ndarray:
    """Soft spectral-ratio mask on delay-and-sum output (NMF prior × DS)."""
    cfg = cfg or DCBConfig()
    X, masks = _init_masks(mix, sr, cfg, mic_pos, doas)
    outs = []
    for k in range(len(masks)):
        _ps, _pn, a_s, _ = _source_stats(X, masks, k, sr, cfg, mic_pos, doas)
        y = apply_beamformer(X, ds_weights(a_s))
        outs.append(masks[k] * y)
    return _synth_all(outs, sr, cfg, mix.shape[0])


def run_all_baselines(
    mix: np.ndarray,
    sr: int,
    mic_pos: np.ndarray | None = None,
    cfg: DCBConfig | None = None,
    doas: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """Run ~10 cocktail-party separators. Returns name -> (n_src, n_samples)."""
    cfg = cfg or DCBConfig()
    results: dict[str, np.ndarray] = {}
    results["NMF"] = separate_nmf_all(mix, sr, mic_pos, cfg, doas)
    results["DS"] = separate_ds_all(mix, sr, mic_pos, cfg, doas)
    results["Ratio+DS"] = separate_ratio_all(mix, sr, mic_pos, cfg, doas)
    results["MPDR"] = separate_mpdr_all(mix, sr, mic_pos, cfg, doas)
    results["MVDR"] = separate_mvdr_all(mix, sr, mic_pos, cfg, doas)
    results["LCMV"] = separate_lcmv_all(mix, sr, mic_pos, cfg, doas)
    results["GEV"] = separate_gev_all(mix, sr, mic_pos, cfg, doas)
    results["MWF"] = separate_mwf_all(mix, sr, mic_pos, cfg, doas)
    results["ZF"] = separate_zf_all(mix, sr, mic_pos, cfg, doas)
    results["MVDR+Mask"] = separate_mvdr_mask_all(mix, sr, mic_pos, cfg, doas)
    results["AuxIVA"] = separate_auxiva_all(mix, sr, mic_pos, cfg, doas)
    results["ILRMA"] = separate_ilrma_all(mix, sr, mic_pos, cfg, doas)
    dcb, _ = dcb_nmf_separate(mix, sr, mic_pos=mic_pos, cfg=cfg, doas=doas)
    results["DCB-NMF"] = dcb
    return results
