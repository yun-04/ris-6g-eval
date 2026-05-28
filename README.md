# ris-6g-eval

Link-level simulation code for the paper:

> Y. Zhao, M. Ji, Z. Li, "A Joint Link-Level Framework for RIS-Assisted
> SISO Systems with Imperfect CSI, Finite Phase Quantization, and
> Deployment Geometry."

The repository reproduces the four figures used in the paper. The
geometry diagram (`fig1_sys`) is rendered by `ris_sim.py`; the three
numerical results in Section IV (`fig_val`, `fig_heatmap`, `fig_design`)
are rendered by `ris_joint_sim.py`. Both scripts share the helpers in
`ris_sim.py` and the matplotlib styling in `plot_style.py`.

## Repository layout
```
.
├── ris_sim.py          # Channel/signal helpers + Fig. 1 (system geometry)
├── ris_joint_sim.py    # Section IV: fig_val / fig_heatmap / fig_design
├── plot_style.py       # IEEE-conference matplotlib styling
└── figures/            # Generated figures (PDF + 600 dpi PNG)
```

## Figures produced
| File           | Script             | Purpose |
|----------------|--------------------|---------|
| `fig1_sys`     | `ris_sim.py`       | System geometry diagram showing the four jointly-modelled parameters {N, b, x_R, sigma_e^2} |
| `fig_val`      | `ris_joint_sim.py` | Validates the joint closed-form rate approximation against Monte-Carlo for ideal (b=inf, sigma_e^2=0) and non-ideal (b=3, sigma_e^2=-9 dB) regimes |
| `fig_heatmap`  | `ris_joint_sim.py` | Achievable-rate contours over the (b, sigma_e^2) plane at N=64 |
| `fig_design`   | `ris_joint_sim.py` | Minimum-N feasibility curves from the joint sizing rule for target rate R_t = 6 bps/Hz |

All outputs are written to `figures/` as both `.pdf` (vector, used in
the paper) and `.png` (600 dpi raster, for previews).

## Requirements
- Python 3.9+
- numpy, scipy, matplotlib

```bash
pip install numpy scipy matplotlib
```

## Reproducing the figures
```bash
python ris_sim.py          # writes figures/fig1_sys.{pdf,png}
python ris_joint_sim.py    # writes fig_val / fig_heatmap / fig_design
```

The Monte-Carlo seed is fixed (`default_rng(2026)`), so two runs on the
same machine produce bit-identical figures.

## License
Released under the MIT License — see [LICENSE](LICENSE).
