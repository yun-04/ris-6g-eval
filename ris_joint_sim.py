"""A Joint Link-Level Framework for Ris-Assisted
Wireless Communication with Imperfect CSI, Finite
Phase Quantization, and Deployment Geometry

Generates the three figures required by the new Section IV (Numerical
Results) of the restructured paper:

    fig_val.pdf      - Validation of the joint closed-form approximation
                       eq. (joint-power) against Monte-Carlo simulation,
                       for the ideal (b=inf, sigma_e^2=0) and a
                       representative non-ideal (b=3, sigma_e^2=-9 dB)
                       regime.
    fig_heatmap.pdf  - Contour map of achievable rate over the
                       (b, sigma_e^2) plane at fixed N=64.
    fig_design.pdf   - Joint minimum-N feasibility curves derived from
                       the sizing rule eq. (size) for the target rate
                       R_t = 6 bps/Hz.

The script reuses the helpers in ris_sim.py and plot_style.py to keep
the visual identity consistent with the rest of the paper.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numpy.random import default_rng

import plot_style
from ris_sim import (
    db2lin, make_channels, phases_optimal, quantize, composite,
    snr_lin, rate, ris_pos, NOISE_DBM,
    ALPHA_D, ALPHA_TI, ALPHA_IR, KdB_TI,
    path_loss_lin,
)

plot_style.apply()
PAL = plot_style.PALETTE
MK = plot_style.MARKERS
LS = plot_style.LINESTYLES

OUT = '../figures'
os.makedirs(OUT, exist_ok=True)


# -------------------------------------------------------------------
# Closed-form helpers: c_b, c_e, and the joint power approximation
# -------------------------------------------------------------------
def c_b(bits):
    """Quantization loss factor c_b = sinc(pi/2^b)."""
    if bits is None or np.isinf(bits):
        return 1.0
    x = np.pi / (2.0 ** bits)
    return np.sin(x) / x


def c_e(sigma_e2_lin):
    """CSI loss factor c_e = 1/(1+sigma_e^2)."""
    return 1.0 / (1.0 + sigma_e2_lin)


def channel_stats(x_R=50.0, K_dB=KdB_TI, n_samples=200000, seed=2027):
    """Estimate the per-element amplitude statistics needed by the
    closed-form approximation.

    Returns
    -------
    beta_d, beta_AI, beta_IU : float
        Large-scale gains.
    m_d : float
        E{|h_d|}.
    A : float
        E{|h_n|} * E{|g_n|}, evaluated by Monte Carlo.
    """
    rng_ = default_rng(seed)
    RIS = ris_pos(x=x_R)
    h_d, h, g = make_channels(1, RIS, n_samples, rng_,
                              K_TI=K_dB, K_IR=K_dB)
    beta_d = path_loss_lin(100.0, ALPHA_D)
    beta_AI = path_loss_lin(np.linalg.norm(RIS - np.array([0.0, 0.0])),
                            ALPHA_TI)
    beta_IU = path_loss_lin(np.linalg.norm(RIS - np.array([100.0, 0.0])),
                            ALPHA_IR)
    m_d = np.mean(np.abs(h_d))
    A = np.mean(np.abs(h[:, 0])) * np.mean(np.abs(g[:, 0]))
    return beta_d, beta_AI, beta_IU, m_d, A


def power_joint_theory(N, bits, sigma_e2_lin, stats):
    """Closed-form joint approximation of E{|h_eff|^2}, eq. (joint-power)."""
    beta_d, beta_AI, beta_IU, m_d, A = stats
    cb = c_b(bits)
    ce = c_e(sigma_e2_lin)
    return (beta_d
            + 2.0 * N * m_d * A * cb * ce
            + N * beta_AI * beta_IU
            + N * (N - 1) * A ** 2 * cb ** 2 * ce ** 2)


def rate_from_power(power, Pt_dBm=20.0):
    """Map an average power to log2(1 + Pt * power / sigma^2)."""
    Pt_W = db2lin(Pt_dBm) * 1e-3
    sigma2 = db2lin(NOISE_DBM) * 1e-3
    return np.log2(1.0 + Pt_W * power / sigma2)


# -------------------------------------------------------------------
# Monte-Carlo helpers (apply quantization + LMMSE-like CSI noise)
# -------------------------------------------------------------------
def _add_csi_noise(h_d, h, g, sigma_e2_lin, rng_):
    if sigma_e2_lin <= 0.0:
        return h_d, h, g
    sH = np.sqrt(sigma_e2_lin * np.mean(np.abs(h) ** 2))
    sG = np.sqrt(sigma_e2_lin * np.mean(np.abs(g) ** 2))
    sD = np.sqrt(sigma_e2_lin * np.mean(np.abs(h_d) ** 2))

    def _rl(shape):
        return ((rng_.standard_normal(shape) + 1j * rng_.standard_normal(shape))
                / np.sqrt(2.0))
    return (h_d + sD * _rl(h_d.shape),
            h + sH * _rl(h.shape),
            g + sG * _rl(g.shape))


def mc_power_and_rate(N, bits, sigma_e2_lin, x_R=50.0,
                      Pt_dBm=20.0, n_trials=4000, seed=42):
    rng_ = default_rng(seed)
    RIS = ris_pos(x=x_R)
    h_d, h, g = make_channels(N, RIS, n_trials, rng_)
    h_d_n, h_n, g_n = _add_csi_noise(h_d, h, g, sigma_e2_lin, rng_)
    theta = phases_optimal(h_d_n, h_n, g_n)
    theta = quantize(theta, bits)
    h_eff = composite(h_d, h, g, theta)
    power = np.mean(np.abs(h_eff) ** 2)
    rate_mc = np.mean(rate(snr_lin(Pt_dBm, h_eff)))
    return power, rate_mc


# ============================================================
# Fig_val: theory vs Monte-Carlo, ideal and non-ideal regimes
# ============================================================
def fig_validation():
    Ns = np.array([8, 16, 32, 64, 96, 128])
    Ns_mc = np.array([8, 16, 32, 64, 96, 128])
    stats = channel_stats(x_R=50.0, K_dB=KdB_TI)
    Pt_dBm = 20.0

    regimes = [
        dict(label='Ideal: $b{=}\\infty,\\,\\sigma_e^{2}{=}0$',
             bits=np.inf, s2=0.0, idx=0),
        dict(label='Non-ideal: $b{=}3,\\,\\sigma_e^{2}{=}-9$ dB',
             bits=3, s2=db2lin(-9.0), idx=1),
    ]

    # Reference baselines (random phases, no RIS)
    rng_ = default_rng(7)
    RIS = ris_pos(x=50.0)
    rate_no_ris = []
    rate_random = []
    for N in Ns_mc:
        h_d, h, g = make_channels(max(N, 1), RIS, 4000, rng_)
        rate_no_ris.append(np.mean(rate(snr_lin(Pt_dBm, h_d))))
        theta = rng_.uniform(0, 2 * np.pi, size=(4000, N))
        rate_random.append(np.mean(rate(snr_lin(
            Pt_dBm, composite(h_d, h, g, theta)))))

    fig, ax = plt.subplots(figsize=(plot_style.COL, 2.95))

    for r in regimes:
        # Theory curve (dense N grid)
        N_dense = np.linspace(Ns.min(), Ns.max(), 80)
        p_th = power_joint_theory(N_dense, r['bits'], r['s2'], stats)
        rate_th = rate_from_power(p_th, Pt_dBm=Pt_dBm)
        ax.plot(N_dense, rate_th,
                color=PAL[r['idx']], linestyle=LS[r['idx']],
                linewidth=1.7, label=f"Theory, {r['label']}")
        # Monte-Carlo markers
        rate_mc = []
        for N in Ns_mc:
            _, rmc = mc_power_and_rate(int(N), r['bits'], r['s2'],
                                       Pt_dBm=Pt_dBm,
                                       n_trials=4000,
                                       seed=100 + r['idx'])
            rate_mc.append(rmc)
        ax.plot(Ns_mc, rate_mc,
                marker=MK[r['idx']], linestyle='None',
                markerfacecolor='white',
                markeredgecolor=PAL[r['idx']], markeredgewidth=1.2,
                markersize=6.5,
                label=f"Sim., {r['label']}")

    # Configuration baselines
    ax.plot(Ns_mc, rate_random,
            marker=MK[3], color=PAL[3], linestyle=LS[3],
            linewidth=1.4, markersize=5.5,
            markeredgecolor='black', markeredgewidth=0.4,
            label='Random RIS')
    ax.plot(Ns_mc, rate_no_ris,
            marker=MK[4], color=PAL[4], linestyle=LS[4],
            linewidth=1.4, markersize=5.5,
            markeredgecolor='black', markeredgewidth=0.4,
            label='No RIS')

    ax.set_xlabel(r'RIS size $N$')
    ax.set_ylabel('Achievable rate (bps/Hz)')
    ax.set_xticks([8, 16, 32, 64, 96, 128])
    ax.legend(loc='lower right', bbox_to_anchor=(1.0, 0.07),
              fontsize=6.8, ncol=1, framealpha=0.88,
              borderpad=0.3, labelspacing=0.22, handlelength=1.6,
              handletextpad=0.45, borderaxespad=0.35)
    plot_style.style_axes(ax)
    plt.tight_layout(pad=0.2)
    plt.savefig(f'{OUT}/fig_val.pdf')
    plt.savefig(f'{OUT}/fig_val.png', dpi=600)
    plt.close()
    print('fig_val saved')


# ============================================================
# Fig_heatmap: rate over (b, sigma_e^2) plane at N=64
# ============================================================
def fig_heatmap():
    bits_grid = np.array([1, 2, 3, 4, 5, 8])  # 8 ~= continuous
    sigma_dB = np.linspace(-30, 0, 16)
    sigma_lin = db2lin(sigma_dB)
    N = 64
    Pt_dBm = 20.0
    stats = channel_stats(x_R=50.0, K_dB=KdB_TI)

    # Theory grid
    R_th = np.zeros((len(sigma_dB), len(bits_grid)))
    for i, s2 in enumerate(sigma_lin):
        for j, b in enumerate(bits_grid):
            p = power_joint_theory(N, b if b < 8 else np.inf, s2, stats)
            R_th[i, j] = rate_from_power(p, Pt_dBm)

    # Monte-Carlo grid (coarser)
    bits_mc = [1, 2, 3, 4, np.inf]
    sigma_mc_dB = np.array([-30, -21, -15, -9, -3, 0])
    R_mc = np.zeros((len(sigma_mc_dB), len(bits_mc)))
    for i, sd in enumerate(sigma_mc_dB):
        s2 = db2lin(sd)
        for j, b in enumerate(bits_mc):
            _, rmc = mc_power_and_rate(N, b, s2,
                                       Pt_dBm=Pt_dBm,
                                       n_trials=2500,
                                       seed=1000 + 7 * i + j)
            R_mc[i, j] = rmc

    fig, ax = plt.subplots(figsize=(plot_style.COL, 3.05))

    # Filled contour from theory
    B_grid, S_grid = np.meshgrid(bits_grid, sigma_dB)
    cs = ax.contourf(B_grid, S_grid, R_th, levels=14, cmap='viridis')
    cb_bar = fig.colorbar(cs, ax=ax, pad=0.02)
    cb_bar.set_label('Achievable rate (bps/Hz)', fontsize=9)
    cb_bar.ax.tick_params(labelsize=8.5)
    cs_lines = ax.contour(B_grid, S_grid, R_th, levels=8,
                          colors='white', linewidths=0.55, alpha=0.65)
    ax.clabel(cs_lines, inline=True, fontsize=7.5, fmt='%.1f')

    # Monte-Carlo overlay
    for j, b in enumerate(bits_mc):
        bx = b if not np.isinf(b) else 8
        for i, sd in enumerate(sigma_mc_dB):
            ax.scatter(bx, sd, marker='o', s=22,
                       facecolor='none',
                       edgecolor='white', linewidth=1.0, zorder=5)

    # Recommended robust-operation envelope
    ax.add_patch(plt.Rectangle((3.0, -30), 5.5, 21.0,
                               fill=False, edgecolor='#FFFFFF',
                               linewidth=1.3, linestyle=(0, (4, 2))))
    ax.text(5.4, -27.5, 'Robust region',
            color='white', fontsize=8.5, ha='center',
            fontweight='bold')

    ax.set_xticks([1, 2, 3, 4, 5, 8])
    ax.set_xticklabels(['1', '2', '3', '4', '5', r'$\infty$'])
    ax.set_xlabel('Phase resolution $b$ (bits)')
    ax.set_ylabel(r'CSI error $\sigma_e^{2}$ (dB)')
    ax.set_xlim(1, 8)
    ax.set_ylim(-30, 0)
    plot_style.style_axes(ax)

    # Sim/Theory legend
    handles = [
        Line2D([0], [0], marker='o', linestyle='None',
               markerfacecolor='none', markeredgecolor='white',
               markeredgewidth=1.2, markersize=6,
               label='Monte-Carlo'),
        Line2D([0], [0], color='white', linewidth=1.0,
               label='Theory contour'),
    ]
    ax.legend(handles=handles, loc='upper right',
              fontsize=8.0, framealpha=0.6, edgecolor='#888')
    plt.tight_layout(pad=0.2)
    plt.savefig(f'{OUT}/fig_heatmap.pdf')
    plt.savefig(f'{OUT}/fig_heatmap.png', dpi=600)
    plt.close()
    print('fig_heatmap saved')


# ============================================================
# Fig_design: joint minimum-N feasibility curves
#   (a) Family A: sigma_e^2 = 0 (perfect CSI), N_min vs b
#   (b) Family B: b = inf      (continuous),   N_min vs sigma_e^2
#   (c) Family C: iso-N_min contours on the (b, sigma_e^2) plane
# ============================================================
def fig_design():
    R_t = 6.0          # target rate (bps/Hz)
    Pt_dBm = 20.0
    Pt_W = db2lin(Pt_dBm) * 1e-3
    sigma2 = db2lin(NOISE_DBM) * 1e-3
    gamma_t = 2.0 ** R_t - 1.0
    target_pow = gamma_t * sigma2 / Pt_W

    stats = channel_stats(x_R=50.0, K_dB=KdB_TI)
    _, _, _, _, A = stats

    # Sizing rule eq.(size): N_min = ceil( sqrt(target_pow) / (A * c_b * c_e) )
    def n_min(b_val, sigma2_lin):
        cb = c_b(b_val)
        ce = c_e(sigma2_lin)
        return np.ceil(np.sqrt(target_pow) / (A * cb * ce))

    fig, axes = plt.subplots(1, 3, figsize=(plot_style.DCOL * 0.98, 2.55),
                             gridspec_kw={'width_ratios': [1.0, 1.0, 1.18]})
    ax1, ax2, ax3 = axes

    # ---- Panel (a): Family A -- perfect CSI, N_min vs b -------------
    bits_axis = np.array([1, 2, 3, 4, 5, 8])
    Nm_A = [n_min(b if b < 8 else np.inf, 0.0) for b in bits_axis]
    ax1.plot(bits_axis, Nm_A, marker=MK[0], color=PAL[0],
             linestyle=LS[0], markersize=5.5,
             label=r'$\sigma_e^{2}=0$')
    ax1.set_xticks([1, 2, 3, 4, 5, 8])
    ax1.set_xticklabels(['1', '2', '3', '4', '5', r'$\infty$'])
    ax1.set_xlabel(r'Phase resolution $b$ (bits)')
    ax1.set_ylabel(r'Minimum $N$')
    ax1.set_title(r'(a) Family A: $\sigma_e^{2}{=}0$', fontsize=9.0)
    ax1.legend(loc='upper right', fontsize=8.0)
    plot_style.style_axes(ax1)

    # ---- Panel (b): Family B -- continuous phase, N_min vs sigma_e^2
    sigma_dB_axis = np.linspace(-30, -3, 28)
    Nm_B = [n_min(np.inf, db2lin(sd)) for sd in sigma_dB_axis]
    ax2.plot(sigma_dB_axis, Nm_B, color=PAL[1],
             linestyle=LS[1], linewidth=1.7,
             label=r'$b=\infty$')
    sd_marks = np.array([-30, -21, -15, -9, -6, -3])
    Nm_B_marks = [n_min(np.inf, db2lin(sd)) for sd in sd_marks]
    ax2.plot(sd_marks, Nm_B_marks, marker=MK[1], color=PAL[1],
             linestyle='None', markerfacecolor='white',
             markeredgecolor=PAL[1], markeredgewidth=1.1,
             markersize=5.5)
    ax2.set_xlabel(r'CSI error $\sigma_e^{2}$ (dB)')
    ax2.set_ylabel(r'Minimum $N$')
    ax2.set_title(r'(b) Family B: $b{=}\infty$', fontsize=9.0)
    ax2.legend(loc='upper left', fontsize=8.0)
    plot_style.style_axes(ax2)

    # ---- Panel (c): Family C -- iso-N_min contours -----------------
    bits_grid = np.array([1, 2, 3, 4, 5, 8])
    sigma_dB = np.linspace(-30, 0, 60)
    BB, SS = np.meshgrid(bits_grid, sigma_dB)
    NM = np.zeros_like(BB, dtype=float)
    for i in range(BB.shape[0]):
        for j in range(BB.shape[1]):
            b_val = BB[i, j] if BB[i, j] < 8 else np.inf
            NM[i, j] = n_min(b_val, db2lin(SS[i, j]))

    levels = [64, 80, 100, 130, 170, 220, 300]
    cs = ax3.contourf(BB, SS, NM, levels=levels, cmap='magma_r',
                      extend='both')
    cb_bar = fig.colorbar(cs, ax=ax3, pad=0.03)
    cb_bar.set_label(r'$N_{\min}$', fontsize=9)
    cb_bar.ax.tick_params(labelsize=8.0)
    cs_l = ax3.contour(BB, SS, NM, levels=levels,
                       colors='white', linewidths=0.55, alpha=0.7)
    ax3.clabel(cs_l, inline=True, fontsize=7.0, fmt='%d')

    # Recommended operating point
    ax3.scatter(3, -9, marker='*', s=150, color='#FFD600',
                edgecolor='black', linewidth=0.8, zorder=6)
    ax3.annotate(r'$(b{=}3,\,\sigma_e^{2}{=}{-}9$ dB$)$',
                 xy=(3, -9), xytext=(3.6, -3.5),
                 color='white', fontsize=7.5, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='white',
                                 linewidth=0.9, shrinkA=2, shrinkB=4))

    ax3.set_xticks([1, 2, 3, 4, 5, 8])
    ax3.set_xticklabels(['1', '2', '3', '4', '5', r'$\infty$'])
    ax3.set_xlabel(r'Phase resolution $b$ (bits)')
    ax3.set_ylabel(r'CSI error $\sigma_e^{2}$ (dB)')
    ax3.set_title(r'(c) Family C: iso-$N_{\min}$', fontsize=9.0)
    plot_style.style_axes(ax3)

    plt.tight_layout(pad=0.3, w_pad=0.8)
    plt.savefig(f'{OUT}/fig_design.pdf')
    plt.savefig(f'{OUT}/fig_design.png', dpi=600)
    plt.close()
    print('fig_design saved')


if __name__ == '__main__':
    import sys
    import time
    t0 = time.time()
    args = set(sys.argv[1:])
    do_all = (not args)
    if do_all or 'val' in args:
        fig_validation()
    if do_all or 'heatmap' in args:
        fig_heatmap()
    if do_all or 'design' in args:
        fig_design()
    print(f'Total: {time.time() - t0:.1f}s')
