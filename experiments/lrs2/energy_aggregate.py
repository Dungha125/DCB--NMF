import sys, json, csv, os, time
from pathlib import Path
from dataclasses import replace
import numpy as np
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.dont_write_bytecode = True
from scipy.io import wavfile
from dcb_nmf.method import DCBConfig, dcb_nmf_separate, separate_mvdr_all
from dcb_nmf.baselines import separate_mvdr_mask_all
from dcb_nmf.metrics import permute_si_sdri
from dcb_nmf.mix import azimuth_to_broadside_doa, mic_x_from_xyz
from dcb_nmf.stft import analysis
SR = 16000; DATA = REPO / "data" / "lrs2_2mix_spatial"
OUT = REPO / "results" / "energy" / "energy_aggregate_rows.jsonl"
CFG = DCBConfig(n_fft=1024, hop=256, n_sources=2, n_outer=3, n_nmf_iter=80, n_fuse_iter=60,
                steering_mode="manifold", seed=0, alpha=0.4)
def load(p):
    sr, raw = wavfile.read(p); y = raw.astype(np.float64)
    if np.issubdtype(raw.dtype, np.integer): y /= np.iinfo(raw.dtype).max
    return y
def fe(y):
    s = np.abs(analysis(y, sr=SR, n_fft=1024, hop=512)); return 10*np.log10((s**2).sum(0)+1e-12)
def scale(e, r): return (e @ r) / (r @ r + 1e-12) * e
done = set()
if OUT.exists():
    for l in OUT.open(): d = json.loads(l); done.add((d["config"], d["file"]))
budget = float(sys.argv[1]) if len(sys.argv) > 1 else float("inf"); t0 = time.time()
for folder in sorted(DATA.glob("config_*")):
    meta = json.loads((folder / "config_metadata.json").read_text())
    ang = meta["angle_config"]; mic = mic_x_from_xyz(meta["mic_positions_xyz"])
    doas = [azimuth_to_broadside_doa(ang["s1_angle"]), azimuth_to_broadside_doa(ang["s2_angle"])]
    for mp in sorted((folder / "mix").glob("*.wav")):
        if (folder.name, mp.name) in done: continue
        if time.time() - t0 > budget: sys.exit(0)
        mix = load(mp); s1 = load(folder / "s1" / mp.name); s2 = load(folder / "s2" / mp.name)
        s1 = s1[:, 0] if s1.ndim == 2 else s1; s2 = s2[:, 0] if s2.ndim == 2 else s2
        n = min(len(mix), len(s1), len(s2)); mix = mix[:n]; refs = np.stack([s1[:n], s2[:n]])
        est = {"MVDR": separate_mvdr_all(mix, SR, mic, CFG, doas),
               "MVDR+Mask": separate_mvdr_mask_all(mix, SR, mic, CFG, doas),
               "DCB-NMF": dcb_nmf_separate(mix, SR, mic_pos=mic, cfg=CFG, doas=doas)[0]}
        row = {"config": folder.name, "sep": ang["sep"], "file": mp.name, "si": {}, "err": {}}
        Er = [fe(refs[0]), fe(refs[1])]
        for name, y in est.items():
            sc, perm, mean = permute_si_sdri(y, refs, mix[:, 0]); row["si"][name] = float(mean)
            per = []
            for k in (0, 1):
                ek = fe(scale(y[perm[k]], refs[k])); er, ei = Er[k], Er[1 - k]
                valid = er > er.max() - 40; weak = valid & (er < ei); strong = valid & (er >= ei)
                d = ek - er
                per.append({c: [float(np.mean(np.abs(d[m]))) if m.any() else None,
                                float(np.mean(d[m])) if m.any() else None, int(m.sum())]
                            for c, m in (("all", valid), ("weak", weak), ("strong", strong))})
            row["err"][name] = per
        with OUT.open("a") as f: f.write(json.dumps(row) + "\n")
print("ALL DONE")
