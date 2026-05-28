# ris-6g-eval

Link-level simulation code for the paper:

> Y. Zhao, M. Ji, Z. Li, "Performance Analysis and Robust Phase
> Configuration of RIS-Assisted Wireless Communications for 6G."

The repository reproduces every numerical figure in the paper from two
self-contained Python scripts. Both scripts share helpers in
`plot_style.py` to keep the visual identity (colours, markers, fonts)
consistent with the manuscript.

## Repository layout
```
.
├── ris_sim.py          # Figs. 2-7 (BER, rate, quantization, geometry, CSI, K-factor)
├── ris_joint_sim.py    # fig_val / fig_heatmap / fig_design (Section IV)
├── plot_style.py       # IEEE-conference matplotlib styling helpers
└── figures/            # Generated figures (PDF + 600 dpi PNG)
```

## What each script produces
`ris_sim.py`

| Figure | Sweep variable | Output |
|--------|----------------|--------|
| Fig. 2 | direct-link SNR vs RIS size N    | BER curves |
| Fig. 3 | transmit power                    | achievable rate |
| Fig. 4 | phase quantization bits b         | rate (grouped lines) |
| Fig. 5 | RIS horizontal position           | received SNR |
| Fig. 6 | CSI estimation error variance     | rate |
| Fig. 7 | RIS size N for varying Rician K   | rate |

`ris_joint_sim.py`

| Output        | Purpose |
|---------------|---------|
| `fig_val`     | Validates the joint closed-form approximation against Monte-Carlo for ideal and non-ideal (b=3, sigma_e^2=-9 dB) regimes |
| `fig_heatmap` | Achievable rate contours over the (b, sigma_e^2) plane at N=64 |
| `fig_design`  | Minimum-N feasibility curves from the sizing rule for target rate R_t = 6 bps/Hz |

All outputs are written to `figures/` as both `.pdf` (vector, for the
paper) and `.png` (600 dpi raster, for previews).

## Requirements
- Python 3.9+
- numpy, scipy, matplotlib

```bash
pip install numpy scipy matplotlib
```

## Reproducing the figures
```bash
python ris_sim.py
python ris_joint_sim.py
```

The Monte-Carlo seed is fixed (`rng = default_rng(2026)`), so two runs
on the same machine produce bit-identical figures.

## License
Released under the MIT License — see [LICENSE](LICENSE).
