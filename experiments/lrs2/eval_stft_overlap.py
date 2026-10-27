from __future__ import annotations

# --- release paths: repo root on sys.path, results written to results/lrs2_overlap ---
import sys as _sys
from pathlib import Path as _Path
_REPO = _Path(__file__).resolve().parents[2]
_sys.path.insert(0, str(_REPO))
_RESULTS = _REPO / "results" / "lrs2_overlap"
_RESULTS.mkdir(parents=True, exist_ok=True)

import argparse
import csv
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import eval_hn as E  # noqa: E402
from dcb_nmf.baselines import separate_mvdr_all  # noqa: E402
from dcb_nmf.method import dcb_nmf_separate  # noqa: E402

# hop -> overlap label
HOPS = {512: "50%", 256: "75%"}
N_FFT = 1024
# same set as Table III (eval_hn.METHODS omits plain MVDR)
METHODS = ("NMF", "MWF", "MVDR", "MVDR+Mask", "Ratio+DS", "DCB-NMF")


def _run(name, mix, mic_pos, doas, cfg):
    if name == "MVDR":
        return separate_mvdr_all(mix, E.SR, mic_pos, cfg, doas)
    return E._run_method(name, mix, mic_pos, doas, cfg)


def _tune_alpha(configs: list[dict], hop: int, n_tune: int) -> tuple[float, dict[float, float]]:
    cfg0 = replace(E._cfg(), n_fft=N_FFT, hop=hop)
    acc: dict[float, list[float]] = {a: [] for a in E.ALPHA_GRID}
    for spec in configs:
        for mix_path in E._list_mixes(spec)[:n_tune]:
            mix, refs, mix_ref = E._load_example(spec, mix_path)
            for a in E.ALPHA_GRID:
                est, _ = dcb_nmf_separate(
                    mix,
                    E.SR,
                    mic_pos=spec["mic_pos"],
                    cfg=replace(cfg0, alpha=float(a)),
                    doas=spec["doas"],
                )
                acc[a].append(E._score(est, refs, mix_ref)[1])
    mean_by_alpha = {a: float(np.mean(v)) for a, v in acc.items()}
    best = max(mean_by_alpha, key=mean_by_alpha.get)
    return float(best), mean_by_alpha


def _eval_hop(
    configs: list[dict],
    hop: int,
    alpha: float,
    limit: int | None,
) -> list[dict]:
    cfg0 = replace(E._cfg(), n_fft=N_FFT, hop=hop, alpha=alpha)
    rows: list[dict] = []
    t0 = time.time()
    for spec in configs:
        files = E._list_mixes(spec)
        if limit is not None:
            files = files[:limit]
        for i, mix_path in enumerate(files, 1):
            mix, refs, mix_ref = E._load_example(spec, mix_path)
            for method in METHODS:
                est = _run(method, mix, spec["mic_pos"], spec["doas"], cfg0)
                _scores, mean = E._score(est, refs, mix_ref)
                rows.append(
                    {
                        "hop": hop,
                        "overlap": HOPS[hop],
                        "config": spec["name"],
                        "sep_deg": f"{spec['sep']:.0f}",
                        "filename": mix_path.name,
                        "method": method,
                        "alpha": f"{alpha:.1f}" if method == "DCB-NMF" else "",
                        "sisdri_mean": f"{mean:.4f}",
                    }
                )
            if i % 20 == 0:
                print(
                    f"  hop={hop} {spec['name']} {i}/{len(files)} "
                    f"({time.time() - t0:.0f}s)",
                    flush=True,
                )
    return rows


def _table(rows: list[dict], out_tex: Path, alphas: dict[int, float]) -> None:
    """Write the LaTeX table comparing 50% and 75% frame overlap."""
    hops = sorted(HOPS, reverse=True)  # 512 then 256 -> 50% then 75%
    stats: dict[tuple[str, int], np.ndarray] = {}
    for method in METHODS:
        for hop in hops:
            vals = np.array(
                [
                    float(r["sisdri_mean"])
                    for r in rows
                    if r["method"] == method and int(r["hop"]) == hop
                ]
            )
            stats[(method, hop)] = vals

    lines = [
        r"\begin{table}[!tb]",
        r"\caption{Effect of STFT frame overlap on the reverberant LRS2-2mix",
        r"spatial benchmark (500 mixtures, 1024-point Hann window). $\Delta$ is",
        r"the paired change from 50\% to 75\% overlap; $p$ from a two-sided",
        r"Wilcoxon signed-rank test over the 500 paired mixtures.}",
        r"\label{tab:overlap_stft}",
        r"\centering",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"\multirow{2}{*}{Method} & \multicolumn{2}{c}{Mean SI-SDRi (dB)} &"
        r" \multirow{2}{*}{$\Delta$ (dB)} \\",
        r"\cmidrule(lr){2-3}",
        r" & 50\% overlap & 75\% overlap & \\",
        r"\midrule",
    ]
    for method in METHODS:
        a = stats[(method, 512)]
        b = stats[(method, 256)]
        n = min(a.size, b.size)
        delta = float(np.mean(b[:n] - a[:n]))
        try:
            p = float(wilcoxon(b[:n], a[:n]).pvalue)
        except ValueError:
            p = float("nan")
        label = rf"\textbf{{{method}}}" if method == "DCB-NMF" else method
        lines.append(
            f"{label} & {a.mean():.2f}$\\pm${a.std():.1f} & "
            f"{b.mean():.2f}$\\pm${b.std():.1f} & "
            f"{delta:+.2f} ($p{_fmt_p(p)}$) \\\\"
        )
    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\vspace{2pt}",
        r"\par\footnotesize $\alpha$ retuned per overlap on the same 10-file tune"
        rf" subset: $\alpha={alphas[512]:.1f}$ at 50\%, $\alpha={alphas[256]:.1f}$"
        r" at 75\%.",
        r"\end{table}",
        "",
    ]
    out_tex.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", out_tex)


def _fmt_p(p: float) -> str:
    if not np.isfinite(p):
        return "=\\mathrm{n/a}"
    if p < 1e-3:
        exp = int(np.floor(np.log10(p)))
        return f"={p / 10 ** exp:.1f}\\!\\times\\!10^{{{exp}}}"
    return f"={p:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default=str(_REPO / "data" / "lrs2_2mix_spatial"))
    ap.add_argument("--out", default=str(_RESULTS))
    ap.add_argument("--tex", default=str(_RESULTS / "tab_overlap_stft_mixture_level.tex"),
                    help="mixture-level LaTeX table (the paper reports p over the 100 speech pairs)")
    ap.add_argument("--n-tune", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None, help="files per config")
    args = ap.parse_args()

    configs = E._discover_configs(Path(args.data))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    alphas: dict[int, float] = {}
    tune_rows: list[dict] = []
    for hop in sorted(HOPS, reverse=True):
        best, mean_by_alpha = _tune_alpha(configs, hop, args.n_tune)
        alphas[hop] = best
        print(f"hop={hop} ({HOPS[hop]} overlap) best alpha={best:.1f}")
        for a, v in mean_by_alpha.items():
            print(f"   a={a:.1f}  {v:6.2f} dB")
            tune_rows.append(
                {"hop": hop, "overlap": HOPS[hop], "alpha": f"{a:.1f}",
                 "sisdri_mean": f"{v:.4f}"}
            )
    E._write_csv(out_dir / "alpha_tune.csv", tune_rows,
                 ["hop", "overlap", "alpha", "sisdri_mean"])

    rows: list[dict] = []
    for hop in sorted(HOPS, reverse=True):
        rows += _eval_hop(configs, hop, alphas[hop], args.limit)

    E._write_csv(
        out_dir / "sisdri_overlap.csv",
        rows,
        ["hop", "overlap", "config", "sep_deg", "filename", "method", "alpha",
         "sisdri_mean"],
    )

    print("\nMean SI-SDRi (dB) by method and frame overlap")
    print("-" * 52)
    for method in METHODS:
        cells = []
        for hop in sorted(HOPS, reverse=True):
            v = np.array([float(r["sisdri_mean"]) for r in rows
                          if r["method"] == method and int(r["hop"]) == hop])
            cells.append(f"{HOPS[hop]}: {v.mean():6.2f}")
        print(f"  {method:<10} " + "   ".join(cells))

    _table(rows, Path(args.tex), alphas)


if __name__ == "__main__":
    main()
