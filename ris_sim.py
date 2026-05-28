"""RIS-Assisted SISO link-level helpers and system-geometry figure.

This module provides the channel and signal-model helpers shared by the
joint validation experiments in ``ris_joint_sim.py``, plus the
system-geometry diagram (``fig1_sys``) used in the paper.

Geometry, path-loss, channel statistics, optimal phasing, b-bit
quantization, and SNR/rate utilities all live here so the joint
experiments can import them without duplication.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy.random import default_rng

import plot_style
plot_style.apply()
PAL = plot_style.PALETTE
MK = plot_style.MARKERS
LS = plot_style.LINESTYLES

rng = default_rng(2026)
OUT = 'figures'
os.makedirs(OUT, exist_ok=True)


# ----------------------------- Helpers ------------------------------
def db2lin(x):
    return 10.0 ** (x / 10.0)


def rayleigh(shape, rng_):
    return (rng_.standard_normal(shape) + 1j * rng_.standard_normal(shape)) \
        / np.sqrt(2.0)


def rician(shape, K_dB, rng_):
    K = db2lin(K_dB)
    return (np.sqrt(K / (K + 1)) * np.ones(shape, dtype=complex)
            + np.sqrt(1.0 / (K + 1)) * rayleigh(shape, rng_))


def path_loss_lin(d, alpha, PL0_dB=-30.0, d0=1.0):
    return db2lin(PL0_dB - 10.0 * alpha * np.log10(d / d0))


TX = np.array([0.0, 0.0])
RX = np.array([100.0, 0.0])


def ris_pos(x=50.0, y=10.0):
    return np.array([x, y])


ALPHA_D, ALPHA_TI, ALPHA_IR = 3.5, 2.2, 2.2
KdB_TI = KdB_IR = 3.0
NOISE_DBM = -94.0


def make_channels(N, RIS, n_trials, rng_, K_TI=None, K_IR=None):
    bd = path_loss_lin(np.linalg.norm(RX - TX), ALPHA_D)
    bTI = path_loss_lin(np.linalg.norm(RIS - TX), ALPHA_TI)
    bIR = path_loss_lin(np.linalg.norm(RX - RIS), ALPHA_IR)
    h_d = np.sqrt(bd) * rayleigh((n_trials,), rng_)
    K1 = KdB_TI if K_TI is None else K_TI
    K2 = KdB_IR if K_IR is None else K_IR
    if K1 == -np.inf:
        h = np.sqrt(bTI) * rayleigh((n_trials, N), rng_)
    else:
        h = np.sqrt(bTI) * rician((n_trials, N), K1, rng_)
    if K2 == -np.inf:
        g = np.sqrt(bIR) * rayleigh((n_trials, N), rng_)
    else:
        g = np.sqrt(bIR) * rician((n_trials, N), K2, rng_)
    return h_d, h, g


def phases_optimal(h_d, h, g):
    return np.angle(h_d)[:, None] - np.angle(h * g)


def quantize(theta, bits):
    if bits is None or np.isinf(bits):
        return theta
    L = 2 ** bits
    step = 2.0 * np.pi / L
    q = np.round(theta / step) * step
    return np.mod(q, 2.0 * np.pi)


def composite(h_d, h, g, theta):
    return h_d + np.sum(h * g * np.exp(1j * theta), axis=1)


def snr_lin(Pt_dBm, h_eff):
    Pt = db2lin(Pt_dBm - 30.0)
    Pn = db2lin(NOISE_DBM - 30.0)
    return Pt * np.abs(h_eff) ** 2 / Pn


def rate(snr):
    return np.log2(1.0 + snr)


def _save(name):
    plt.savefig(f'{OUT}/{name}.pdf')
    plt.savefig(f'{OUT}/{name}.png', dpi=600)
    plt.close()


# ============================================================
# Fig.1  System geometry diagram
# ============================================================
def fig1_geometry():
    """System geometry visualising the four jointly-modelled parameters
    of Section II:
      (i)   RIS size N            -> labelled on the array
      (ii)  phase resolution b    -> discrete phase wheel beside RIS
      (iii) deployment x_R        -> horizontal arrow on the ground
      (iv)  CSI error variance    -> noisy-estimate annotation
    """
    fig, ax = plt.subplots(figsize=(plot_style.DCOL * 0.78, 3.55))

    AP_X, UE_X = 0.0, 10.0
    RIS_X, RIS_Y = 5.0, 2.55
    n_cols, n_rows = 8, 2
    cell = 0.22

    bbox_h = dict(boxstyle='round,pad=0.20', facecolor='white',
                  edgecolor=PAL[0], linewidth=0.7, alpha=0.95)
    bbox_g = dict(boxstyle='round,pad=0.20', facecolor='white',
                  edgecolor=PAL[2], linewidth=0.7, alpha=0.95)

    ax.annotate('', xy=(UE_X - 0.05, 0.05), xytext=(AP_X + 0.05, 0.05),
                arrowprops=dict(arrowstyle='-', color='#7A7A7A',
                                linestyle=(0, (4, 2.5)), lw=1.0))
    ax.plot(5, 0.05, marker='x', color=PAL[1], markersize=12, mew=2.0,
            zorder=6)
    ax.text(5, -0.55, r'$h_d$  (blocked, NLoS)', ha='center',
            fontsize=8.5, color='#555', style='italic')

    ax.plot(AP_X, 0, marker='^', color=PAL[0], markersize=13, zorder=5,
            markeredgecolor='black', markeredgewidth=0.7)
    ax.text(AP_X, -1.55, 'AP\n(0,0)', ha='center', fontsize=9,
            fontweight='bold')

    ax.plot(UE_X, 0, marker='o', color=PAL[2], markersize=11, zorder=5,
            markeredgecolor='black', markeredgewidth=0.7)
    ax.text(UE_X, -1.55, 'UE\n(100,0) m', ha='center', fontsize=9,
            fontweight='bold')

    arr_left = RIS_X - n_cols * cell / 2
    arr_right = RIS_X + n_cols * cell / 2
    arr_top = RIS_Y + n_rows * cell
    for i in range(n_cols):
        for j in range(n_rows):
            rect = mpatches.Rectangle(
                (arr_left + i * cell, RIS_Y + j * cell),
                cell * 0.9, cell * 0.9,
                facecolor='#F5C45E', edgecolor='black', linewidth=0.55)
            ax.add_patch(rect)
    ax.text(RIS_X, arr_top + 0.30, r'RIS:  $N$ elements',
            ha='center', fontsize=9, fontweight='bold')

    ax.annotate('', xy=(arr_left + cell * 0.5, RIS_Y),
                xytext=(AP_X + 0.30, 0.30),
                arrowprops=dict(arrowstyle='->', color=PAL[0], lw=1.5,
                                connectionstyle='arc3,rad=-0.08'))
    ax.text(1.55, 1.85, r'$\hat{\mathbf{h}}=\mathbf{h}+\mathbf{e}_h$',
            fontsize=9, color=PAL[0], bbox=bbox_h)

    ax.annotate('', xy=(UE_X - 0.30, 0.30),
                xytext=(arr_right - cell * 0.5, RIS_Y),
                arrowprops=dict(arrowstyle='->', color=PAL[2], lw=1.5,
                                connectionstyle='arc3,rad=-0.08'))
    ax.text(8.45, 1.85, r'$\hat{\mathbf{g}}=\mathbf{g}+\mathbf{e}_g$',
            fontsize=9, color=PAL[2], bbox=bbox_g)

    ax.text(RIS_X, 1.65,
            r'$\mathbf{e}_h,\mathbf{e}_g\!\sim\!\mathcal{CN}(\mathbf{0},'
            r'\sigma_e^{2}\mathbf{I})$',
            ha='center', fontsize=8.5, color='#444',
            bbox=dict(boxstyle='round,pad=0.22', facecolor='#F7F2E6',
                      edgecolor='#B98A2E', linewidth=0.6, alpha=0.95))

    y_xr = arr_top + 0.95
    ax.annotate('', xy=(RIS_X, y_xr), xytext=(0, y_xr),
                arrowprops=dict(arrowstyle='<->', color='#1A4F8B', lw=0.9))
    ax.text(RIS_X / 2, y_xr + 0.18, r'$x_R$',
            ha='center', fontsize=9.5, color='#1A4F8B',
            fontweight='bold')
    ax.plot([RIS_X, RIS_X], [0, RIS_Y], color='#1A4F8B', lw=0.7,
            ls=(0, (2, 2)))
    ax.annotate('', xy=(RIS_X + 0.55, RIS_Y), xytext=(RIS_X + 0.55, 0),
                arrowprops=dict(arrowstyle='<->', color='#1A4F8B', lw=0.7))
    ax.text(RIS_X + 0.78, RIS_Y / 2, r'$h_R$',
            fontsize=8.5, color='#1A4F8B', va='center')

    wheel_cx, wheel_cy, R = 11.55, 2.85, 0.55
    circ = mpatches.Circle((wheel_cx, wheel_cy), R, fill=False,
                           edgecolor='#444', lw=0.9)
    ax.add_patch(circ)
    L = 8
    for k in range(L):
        ang = 2 * np.pi * k / L
        x0, y0 = wheel_cx + R * np.cos(ang), wheel_cy + R * np.sin(ang)
        ax.plot(x0, y0, marker='o', color='#E07B00',
                markersize=4.0, markeredgecolor='black',
                markeredgewidth=0.4, zorder=5)
    ang = 2 * np.pi * 1 / L
    ax.annotate('', xy=(wheel_cx + R * np.cos(ang),
                        wheel_cy + R * np.sin(ang)),
                xytext=(wheel_cx, wheel_cy),
                arrowprops=dict(arrowstyle='->', color='#222', lw=0.9))
    ax.text(wheel_cx, wheel_cy + R + 0.28,
            r'$\theta_n\!\in\!\{2\pi k/2^{b}\}$',
            ha='center', fontsize=8.5)
    ax.text(wheel_cx, wheel_cy - R - 0.32, r'$b$-bit quantizer',
            ha='center', fontsize=8.5, color='#7A4500',
            fontweight='bold')
    ax.annotate('', xy=(wheel_cx - R - 0.04, wheel_cy),
                xytext=(arr_right + 0.04, RIS_Y + cell * n_rows / 2),
                arrowprops=dict(arrowstyle='-', color='#7A7A7A',
                                lw=0.6, ls=(0, (2, 2))))

    ax.annotate('', xy=(UE_X, -2.45), xytext=(0, -2.45),
                arrowprops=dict(arrowstyle='<->', color='black', lw=0.8))
    ax.text(5, -2.85, r'$d_{\mathrm{AU}} = 100$ m',
            ha='center', fontsize=8.5)

    ax.text(RIS_X, y_xr + 1.05,
            r'Joint parameters: $\{N,\ b,\ x_R,\ \sigma_e^{2}\}$',
            ha='center', fontsize=9, fontweight='bold',
            color='#222', bbox=dict(boxstyle='round,pad=0.30',
                                    facecolor='#F4F4F4',
                                    edgecolor='#888', lw=0.6))

    ax.set_xlim(-1.4, 12.8)
    ax.set_ylim(-3.4, 5.4)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.grid(False)
    plt.tight_layout(pad=0.15)
    _save('fig1_sys')
    print('Fig.1 (geometry) saved')


if __name__ == '__main__':
    fig1_geometry()
