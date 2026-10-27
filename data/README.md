# LRS2-2mix spatial benchmark (Sec. IV-B)

`lrs2_2mix_spatial/` holds the reverberant two-microphone spatializations used in
Table III, Table IV and Fig. 3 of the paper.

```
lrs2_2mix_spatial/
  simulation_metadata.json      room, array and image-method parameters
  top100.csv, top100_manifest.csv   the 100 LRS2-2mix test mixtures used, with the
                                selection score (SI-SDRi of a reference separation model)
  config_0X_sepYYY_s1_AAA_s2_BBB/   one folder per angle configuration (5 in total)
    config_metadata.json        source azimuths, mic positions
    mix/*.wav                   2-channel mixture (16 kHz)
    s1/*.wav, s2/*.wav          reverberant spatial images (reference at mic 0)
    ref_clean/s1, ref_clean/s2  dry source signals (not used for scoring)
```

* Source: LRS2-2mix test split (`mix_2_spk_min_tt`), 100 mixtures with the highest
  SI-SDRi of a reference single-channel separation model (18.0–21.2 dB).
* Room: 3 × 3 × 2.5 m shoebox, image method of order 2, reflection coefficient
  0.55, c = 343 m/s, RIR length 0.25 s.
* Array: two microphones, 4.57 cm spacing, centred at (1.5, 1.5, 1.25) m; talkers
  at 1 m. Azimuths (array axis = 0°): 15°/165°, 30°/150°, 45°/135°, 60°/120°,
  75°/105° (separations 150°, 120°, 90°, 60°, 30°).
* The same 100 speech pairs appear under all five configurations (500 mixtures).
  The α-tune subset is the first two files of each configuration (2 speech pairs).

## Licence of the audio

The audio is derived from the **LRS2 dataset (BBC)**, which is distributed for
non-commercial academic research only under the LRS2 data-sharing agreement.
The `.wav` files are therefore excluded from the public repository by
`.gitignore`; only the metadata and manifests above are published. To obtain
the audio, request LRS2 access from the dataset owners and contact the
authors for the spatialization, or regenerate it from the manifests and
`simulation_metadata.json`.
