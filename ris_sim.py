"""RIS-Assisted 6G Wireless Communications: paper-quality figures.

All seven figures are regenerated with a unified IEEE-conference style
(see plot_style.py).  The numerical content is unchanged with respect to
the previous revision; only visual presentation has been improved to
address reviewer comments on font sizes, line widths, grid density,
colour palette, and DPI.

Figures
-------
Fig.1 - System geometry diagram (schematic)
Fig.2 - BER vs direct-link SNR for varying RIS size N
Fig.3 - Achievable rate vs transmit power
Fig.4 - Rate vs phase quantization bits (grouped lines, not bars)
Fig.5 - Received SNR vs RIS horizontal position
Fig.6 - Rate vs CSI estimation error variance
Fig.7 - Rate vs N for varying Rician K-factor
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from numpy.random import default_rng
from scipy.special import erfc

import plot_style
plot_style.apply()
PAL = plot_style.PALETTE
MK = plot_style.MARKERS
LS = plot_style.LINESTYLES

rng = default_rng(2026)
OUT = '../figures'
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


def phases_random(N, n_trials, rng_):
    return rng_.uniform(0, 2 * np.pi, size=(n_trials, N))


def quantize(theta, bits):
    if bits is None or bits == np.inf:
        return theta
    L = 2 ** bits
    step = 2 * np.pi / L
    return np.mod(np.round(np.mod(theta, 2 * np.pi) / step) * step, 2 * np.pi)


def composite(h_d, h, g, theta):
    return h_d + np.sum(h * np.exp(1j * theta) * g, axis=1)


def snr_lin(Pt_dBm, h_eff):
    Pt = db2lin(Pt_dBm) * 1e-3
    return Pt * np.abs(h_eff) ** 2 / (db2lin(NOISE_DBM) * 1e-3)


def rate(snr):
    return np.log2(1.0 + snr)


def bpsk_ber(snr):
    return 0.5 * erfc(np.sqrt(snr))


def _save(name):
    plt.savefig(f'{OUT}/{name}.pdf')
    plt.savefig(f'{OUT}/{name}.png', dpi=600)
    plt.close()


# ============================================================
# Fig.1  System geometry diagram
# ============================================================
def fig1_geometry():
    """System geometry that explicitly visualises the four design
    parameters jointly modelled in Section II:
      (i)   RIS size N            -> labelled on the array
      (ii)  phase resolution b    -> discrete phase wheel beside RIS
      (iii) deployment x_R        -> horizontal arrow on the ground
      (iv)  CSI error variance σ_e^2 -> noisy-estimate annotation
    """
    fig, ax = plt.subplots(figsize=(plot_style.DCOL * 0.78, 3.55))

    # Layout coordinates ----------------------------------------------
    AP_X, UE_X = 0.0, 10.0
    RIS_X, RIS_Y = 5.0, 2.55
    n_cols, n_rows = 8, 2
    cell = 0.22

    # White-fill helper for callout boxes (formula labels)
    bbox_h = dict(boxstyle='round,pad=0.20', facecolor='white',
                  edgecolor=PAL[0], linewidth=0.7, alpha=0.95)
    bbox_g = dict(boxstyle='round,pad=0.20', facecolor='white',
                  edgecolor=PAL[2], linewidth=0.7, alpha=0.95)

    # ----- direct path (blocked) --------------------------------------
    ax.annotate('', xy=(UE_X - 0.05, 0.05), xytext=(AP_X + 0.05, 0.05),
                arrowprops=dict(arrowstyle='-', color='#7A7A7A',
                                linestyle=(0, (4, 2.5)), lw=1.0))
    ax.plot(5, 0.05, marker='x', color=PAL[1], markersize=12, mew=2.0,
            zorder=6)
    ax.text(5, -0.55, r'$h_d$  (blocked, NLoS)', ha='center',
            fontsize=8.5, color='#555', style='italic')

    # ----- AP / UE markers --------------------------------------------
    ax.plot(AP_X, 0, marker='^', color=PAL[0], markersize=13, zorder=5,
            markeredgecolor='black', markeredgewidth=0.7)
    ax.text(AP_X, -1.55, 'AP\n(0,0)', ha='center', fontsize=9,
            fontweight='bold')

    ax.plot(UE_X, 0, marker='o', color=PAL[2], markersize=11, zorder=5,
            markeredgecolor='black', markeredgewidth=0.7)
    ax.text(UE_X, -1.55, 'UE\n(100,0) m', ha='center', fontsize=9,
            fontweight='bold')

    # ----- RIS array (parameter N) ------------------------------------
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
    ax.text(RIS_X, arr_top + 0.30,
            r'RIS:  $N$ elements',
            ha='center', fontsize=9, fontweight='bold')

    # ----- AP -> RIS  link (curved, with offset CSI annotation) -------
    ax.annotate('', xy=(arr_left + cell * 0.5, RIS_Y),
                xytext=(AP_X + 0.30, 0.30),
                arrowprops=dict(arrowstyle='->', color=PAL[0], lw=1.5,
                                connectionstyle='arc3,rad=-0.08'))
    ax.text(1.55, 1.85,
            r'$\hat{\mathbf{h}}=\mathbf{h}+\mathbf{e}_h$',
            fontsize=9, color=PAL[0], bbox=bbox_h)

    # ----- RIS -> UE  link (curved, with offset annotation) -----------
    ax.annotate('', xy=(UE_X - 0.30, 0.30),
                xytext=(arr_right - cell * 0.5, RIS_Y),
                arrowprops=dict(arrowstyle='->', color=PAL[2], lw=1.5,
                                connectionstyle='arc3,rad=-0.08'))
    ax.text(8.45, 1.85,
            r'$\hat{\mathbf{g}}=\mathbf{g}+\mathbf{e}_g$',
            fontsize=9, color=PAL[2], bbox=bbox_g)

    # ----- CSI error variance annotation (single bubble below RIS) ----
    ax.text(RIS_X, 1.65,
            r'$\mathbf{e}_h,\mathbf{e}_g\!\sim\!\mathcal{CN}(\mathbf{0},'
            r'\sigma_e^{2}\mathbf{I})$',
            ha='center', fontsize=8.5, color='#444',
            bbox=dict(boxstyle='round,pad=0.22', facecolor='#F7F2E6',
                      edgecolor='#B98A2E', linewidth=0.6, alpha=0.95))

    # ----- horizontal x_R bar (top of figure, clear of RIS label) -----
    y_xr = arr_top + 0.95
    ax.annotate('', xy=(RIS_X, y_xr), xytext=(0, y_xr),
                arrowprops=dict(arrowstyle='<->', color='#1A4F8B', lw=0.9))
    ax.text(RIS_X / 2, y_xr + 0.18, r'$x_R$',
            ha='center', fontsize=9.5, color='#1A4F8B',
            fontweight='bold')
    # vertical drop-line and h_R bar (right side of RIS)
    ax.plot([RIS_X, RIS_X], [0, RIS_Y], color='#1A4F8B', lw=0.7,
            ls=(0, (2, 2)))
    ax.annotate('', xy=(RIS_X + 0.55, RIS_Y), xytext=(RIS_X + 0.55, 0),
                arrowprops=dict(arrowstyle='<->', color='#1A4F8B', lw=0.7))
    ax.text(RIS_X + 0.78, RIS_Y / 2, r'$h_R$',
            fontsize=8.5, color='#1A4F8B', va='center')

    # ----- discrete phase wheel  (parameter b) ------------------------
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
    # active phase pointer
    ang = 2 * np.pi * 1 / L
    ax.annotate('', xy=(wheel_cx + R * np.cos(ang),
                        wheel_cy + R * np.sin(ang)),
                xytext=(wheel_cx, wheel_cy),
                arrowprops=dict(arrowstyle='->', color='#222', lw=0.9))
    ax.text(wheel_cx, wheel_cy + R + 0.28,
            r'$\theta_n\!\in\!\{2\pi k/2^{b}\}$',
            ha='center', fontsize=8.5)
    ax.text(wheel_cx, wheel_cy - R - 0.32,
            r'$b$-bit quantizer',
            ha='center', fontsize=8.5, color='#7A4500',
            fontweight='bold')
    # connector wheel <-> RIS
    ax.annotate('', xy=(wheel_cx - R - 0.04, wheel_cy),
                xytext=(arr_right + 0.04, RIS_Y + cell * n_rows / 2),
                arrowprops=dict(arrowstyle='-', color='#7A7A7A',
                                lw=0.6, ls=(0, (2, 2))))

    # ----- AP-UE distance bar -----------------------------------------
    ax.annotate('', xy=(UE_X, -2.45), xytext=(0, -2.45),
                arrowprops=dict(arrowstyle='<->', color='black', lw=0.8))
    ax.text(5, -2.85, r'$d_{\mathrm{AU}} = 100$ m',
            ha='center', fontsize=8.5)

    # ----- Top header: Joint parameters -------------------------------
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


# ============================================================
# Fig.2  BER vs SNR
# ============================================================
def fig2_ber():
    SNR_dB = np.arange(-10, 21, 2)
    Ns = [0, 16, 32, 64, 128]
    n_trials = 20000
    RIS = ris_pos()

    fig, ax = plt.subplots(figsize=(plot_style.COL, 2.85))
    for idx, N in enumerate(Ns):
        bers = []
        for snr_db in SNR_dB:
            n_t = n_trials if N <= 64 else max(10000, n_trials // 2)
            h_d, h, g = make_channels(max(N, 1), RIS, n_t, rng)
            if N == 0:
                h_eff = h_d
            else:
                theta = phases_optimal(h_d, h, g)
                h_eff = composite(h_d, h, g, theta)
            Pt_W = (db2lin(snr_db) * db2lin(NOISE_DBM) * 1e-3
                    / np.mean(np.abs(h_d) ** 2))
            sigma2 = db2lin(NOISE_DBM) * 1e-3
            inst_snr = Pt_W * np.abs(h_eff) ** 2 / sigma2
            bers.append(np.mean(bpsk_ber(inst_snr)))
        bers = np.clip(np.array(bers), 1e-10, 1.0)
        label = 'No RIS' if N == 0 else fr'RIS, $N={N}$'
        ax.semilogy(SNR_dB, bers,
                    marker=MK[idx], color=PAL[idx],
                    linestyle=LS[idx % len(LS)],
                    label=label, markevery=2,
                    markersize=5.5, linewidth=1.6,
                    markeredgecolor='black', markeredgewidth=0.4)

    # Engineering reference lines
    ax.axhline(1e-3, color='#555555', lw=0.8, ls=(0, (4, 2)), alpha=0.85)
    ax.text(20, 1.4e-3, 'eMBB target', fontsize=8, color='#333',
            ha='right', va='bottom')
    ax.axhline(1e-8, color='#555555', lw=0.8, ls=(0, (4, 2)), alpha=0.85)
    ax.text(20, 1.4e-8, 'URLLC target', fontsize=8, color='#333',
            ha='right', va='bottom')

    # 5-orders-of-magnitude annotation at SNR = 10 dB
    ax.axvline(10, color='#777777', lw=0.7, ls=':', alpha=0.7)
    ax.annotate(r'$\approx 5$ orders',
                xy=(10, 1e-5), xytext=(-7, 1e-7),
                fontsize=8.5, ha='left', color='#1A1A1A',
                arrowprops=dict(arrowstyle='->', color='#333', lw=0.8))

    ax.set_xlabel('Direct-link SNR (dB)')
    ax.set_ylabel('Average BER')
    ax.set_xlim(-10, 20)
    ax.set_ylim(1e-9, 1)
    ax.legend(loc='lower left', ncol=2)
    plot_style.style_axes(ax)
    plt.tight_layout(pad=0.2)
    _save('fig2_ber')
    print('Fig.2 (BER) saved')


# ============================================================
# Fig.3  Rate vs transmit power
# ============================================================
def fig3_rate():
    Pt_dBm = np.arange(-10, 41, 5)
    n_trials = 3000
    RIS = ris_pos()
    schemes = [
        ('No RIS',                0,  'no'),
        ('Random RIS, $N=64$',    64, 'rand'),
        ('Optimal RIS, $N=32$',   32, 'opt'),
        ('Optimal RIS, $N=64$',   64, 'opt'),
    ]
    fig, ax = plt.subplots(figsize=(plot_style.COL, 2.85))
    for k, (label, N, mode) in enumerate(schemes):
        h_d, h, g = make_channels(max(N, 1), RIS, n_trials, rng)
        rates = []
        for p in Pt_dBm:
            if mode == 'no':
                h_eff = h_d
            elif mode == 'rand':
                theta = phases_random(N, n_trials, rng)
                h_eff = composite(h_d, h, g, theta)
            else:
                theta = phases_optimal(h_d, h, g)
                h_eff = composite(h_d, h, g, theta)
            rates.append(np.mean(rate(snr_lin(p, h_eff))))
        ax.plot(Pt_dBm, rates,
                marker=MK[k], color=PAL[k],
                linestyle=LS[k % len(LS)],
                label=label, markersize=5.5, linewidth=1.6,
                markeredgecolor='black', markeredgewidth=0.4)

    ax.axvline(20, color='#777777', lw=0.7, ls=':', alpha=0.7)
    ax.text(20.6, 0.45, r'$P_t = 20$ dBm', fontsize=8.5, color='#333')

    ax.set_xlabel('Transmit power $P_t$ (dBm)')
    ax.set_ylabel('Achievable rate (bps/Hz)')
    ax.legend(loc='upper left')
    plot_style.style_axes(ax)
    plt.tight_layout(pad=0.2)
    _save('fig3_rate')
    print('Fig.3 (rate) saved')


# ============================================================
# Fig.4  Rate vs phase resolution  (replaces grouped bars)
# ============================================================
def fig4_quant():
    bits = [1, 2, 3, 4, np.inf]
    Ns = [16, 32, 64, 128]
    Pt = 20.0
    n_trials = 3000
    RIS = ris_pos()

    fig, ax = plt.subplots(figsize=(plot_style.COL, 2.85))
    x_positions = np.arange(len(bits))
    for j, N in enumerate(Ns):
        h_d, h, g = make_channels(N, RIS, n_trials, rng)
        opt = phases_optimal(h_d, h, g)
        rates = []
        for b in bits:
            tq = quantize(opt, b)
            rates.append(np.mean(rate(snr_lin(Pt, composite(h_d, h, g, tq)))))
        ax.plot(x_positions, rates,
                marker=MK[j], color=PAL[j],
                linestyle=LS[j % len(LS)],
                label=fr'$N={N}$',
                markersize=6.0, linewidth=1.7,
                markeredgecolor='black', markeredgewidth=0.5)

    # Highlight the b=3 region: knee of the curve
    ax.axvspan(1.85, 2.15, color='#888888', alpha=0.10)
    ax.text(2.0, ax.get_ylim()[0] + 0.08, '3-bit knee',
            fontsize=7.8, color='#333', ha='center')

    ax.set_xticks(x_positions)
    ax.set_xticklabels(['1', '2', '3', '4', r'$\infty$'])
    ax.set_xlabel('Phase resolution $b$ (bits)')
    ax.set_ylabel('Achievable rate (bps/Hz)')
    ax.legend(loc='lower right', ncol=2)
    plot_style.style_axes(ax)
    plt.tight_layout(pad=0.2)
    _save('fig4_quant')
    print('Fig.4 (quantization) saved')


# ============================================================
# Fig.5  Received SNR vs RIS position
# ============================================================
def fig5_position():
    xs = np.arange(10, 96, 5)
    Pt = 20.0
    N = 64
    n_trials = 1500

    fig, ax = plt.subplots(figsize=(plot_style.COL, 2.85))
    snr_opt, snr_rand, snr_no = [], [], []
    for x in xs:
        RIS = ris_pos(x=x)
        h_d, h, g = make_channels(N, RIS, n_trials, rng)
        snr_no.append(10 * np.log10(np.mean(snr_lin(Pt, h_d))))
        theta = phases_optimal(h_d, h, g)
        snr_opt.append(10 * np.log10(np.mean(snr_lin(Pt, composite(h_d, h, g, theta)))))
        theta = phases_random(N, n_trials, rng)
        snr_rand.append(10 * np.log10(np.mean(snr_lin(Pt, composite(h_d, h, g, theta)))))

    ax.plot(xs, snr_opt, marker=MK[0], color=PAL[0], linestyle=LS[0],
            label='Optimal RIS', markersize=5.5, linewidth=1.7,
            markeredgecolor='black', markeredgewidth=0.4)
    ax.plot(xs, snr_rand, marker=MK[1], color=PAL[1], linestyle=LS[1],
            label='Random RIS', markersize=5.5, linewidth=1.7,
            markeredgecolor='black', markeredgewidth=0.4)
    ax.plot(xs, snr_no, marker=MK[2], color=PAL[2], linestyle=LS[2],
            label='No RIS', markersize=5.5, linewidth=1.7,
            markeredgecolor='black', markeredgewidth=0.4)

    # Favourable deployment regions
    ax.axvspan(10, 25, color=PAL[0], alpha=0.08)
    ax.axvspan(80, 95, color=PAL[0], alpha=0.08)
    y_top = max(snr_opt) - 1.6
    ax.text(17, y_top, 'Near AP', fontsize=8, ha='center', color=PAL[0])
    ax.text(87, y_top, 'Near UE', fontsize=8, ha='center', color=PAL[0])

    ax.set_xlabel(r'RIS horizontal position $x_R$ (m)')
    ax.set_ylabel('Average received SNR (dB)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.27),
              ncol=3, frameon=False, fontsize=8.5)
    plot_style.style_axes(ax)
    plt.tight_layout(pad=0.2)
    _save('fig5_pos')
    print('Fig.5 (position) saved')


# ============================================================
# Fig.6  Rate vs CSI error
# ============================================================
def fig6_csi():
    sigma_e2_dB = np.arange(-30, 1, 3)
    Ns = [32, 64, 128]
    Pt = 20.0
    n_trials = 3000
    RIS = ris_pos()

    fig, ax = plt.subplots(figsize=(plot_style.COL, 2.95))
    for j, N in enumerate(Ns):
        h_d, h, g = make_channels(N, RIS, n_trials, rng)
        theta_id = phases_optimal(h_d, h, g)
        rate_id = np.mean(rate(snr_lin(Pt, composite(h_d, h, g, theta_id))))
        rates = []
        for s_dB in sigma_e2_dB:
            s2 = db2lin(s_dB)
            sH = np.sqrt(s2 * np.mean(np.abs(h) ** 2))
            sG = np.sqrt(s2 * np.mean(np.abs(g) ** 2))
            sD = np.sqrt(s2 * np.mean(np.abs(h_d) ** 2))
            h_hat = h + sH * rayleigh(h.shape, rng)
            g_hat = g + sG * rayleigh(g.shape, rng)
            d_hat = h_d + sD * rayleigh(h_d.shape, rng)
            theta = phases_optimal(d_hat, h_hat, g_hat)
            rates.append(np.mean(rate(snr_lin(Pt, composite(h_d, h, g, theta)))))
        ax.plot(sigma_e2_dB, rates,
                marker=MK[j], color=PAL[j],
                linestyle=LS[j % len(LS)],
                label=fr'$N={N}$',
                markersize=5.5, linewidth=1.7,
                markeredgecolor='black', markeredgewidth=0.4)
        ax.axhline(rate_id, color=PAL[j], lw=0.7, ls=(0, (4, 2)), alpha=0.55)

    # Near-perfect-CSI band
    ax.fill_betweenx([4.0, 8.6], -30, -15, color='#9DC183', alpha=0.10)
    ax.text(-22.5, 4.15, 'Near-perfect-CSI', fontsize=7.8,
            color='#3E6826', ha='center')

    # -9 dB engineering threshold (label placed at the bottom, away from curves)
    ax.axvline(-9, color='#B22222', lw=1.0, ls=':', alpha=0.9)
    ax.text(-8.6, 4.55, r'$\sigma_e^2 = -9$ dB',
            fontsize=7.8, color='#B22222', ha='left', va='bottom')

    # Annotation band placed in the headroom above all curves
    ax.annotate('', xy=(-9.5, 8.18), xytext=(-29.5, 8.18),
                arrowprops=dict(arrowstyle='->', color='#3E6826', lw=0.9))
    ax.text(-19.5, 8.34, r'$<\!10\%$ rate loss', fontsize=7.8,
            color='#3E6826', ha='center', va='bottom')

    ax.annotate('', xy=(-0.2, 8.18), xytext=(-8.5, 8.18),
                arrowprops=dict(arrowstyle='->', color='#B22222', lw=0.9))
    ax.text(-4.4, 8.34, r'$>\!10\%$ rate loss', fontsize=7.8,
            color='#B22222', ha='center', va='bottom')

    # Custom legend includes the dashed perfect-CSI bound
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Line2D([0], [0], color='#444', lw=0.8,
                          ls=(0, (4, 2)), label='Perfect-CSI bound'))
    ax.legend(handles=handles,
              loc='upper center', bbox_to_anchor=(0.5, -0.27),
              ncol=4, frameon=False, fontsize=8.0)

    ax.set_xlabel(r'CSI estimation error $\sigma_e^2$ (dB)')
    ax.set_ylabel('Achievable rate (bps/Hz)')
    ax.set_ylim(4.0, 8.6)
    plot_style.style_axes(ax)
    plt.tight_layout(pad=0.2)
    _save('fig6_csi')
    print('Fig.6 (CSI) saved')


# ============================================================
# Fig.7  Rate vs N for varying Rician K-factor
# ============================================================
def fig7_kfactor():
    Ns = np.array([8, 16, 32, 64, 128])
    Ks_dB = [-np.inf, 0, 3, 10, 30]
    K_labels = ['Rayleigh', r'$K=0$ dB', r'$K=3$ dB',
                r'$K=10$ dB', r'$K=30$ dB (near-LoS)']
    Pt = 20.0
    n_trials = 4000
    RIS = ris_pos()

    fig, ax = plt.subplots(figsize=(plot_style.COL, 2.85))
    for j, K_dB in enumerate(Ks_dB):
        rates = []
        for N in Ns:
            h_d, h, g = make_channels(int(N), RIS, n_trials, rng,
                                      K_TI=K_dB, K_IR=K_dB)
            theta = phases_optimal(h_d, h, g)
            h_eff = composite(h_d, h, g, theta)
            rates.append(np.mean(rate(snr_lin(Pt, h_eff))))
        ax.plot(Ns, rates,
                marker=MK[j], color=PAL[j],
                linestyle=LS[j % len(LS)],
                label=K_labels[j],
                markersize=5.5, linewidth=1.7,
                markeredgecolor='black', markeredgewidth=0.4)

    ax.set_xscale('log', base=2)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(int(n)) for n in Ns])
    ax.set_xlabel(r'RIS size $N$')
    ax.set_ylabel('Achievable rate (bps/Hz)')
    ax.legend(loc='upper left', fontsize=8.0)
    plot_style.style_axes(ax)
    plt.tight_layout(pad=0.2)
    _save('fig7_kfactor')
    print('Fig.7 (K-factor) saved')


if __name__ == '__main__':
    import time
    t0 = time.time()
    fig1_geometry()
    fig2_ber()
    fig3_rate()
    fig4_quant()
    fig5_position()
    fig6_csi()
    fig7_kfactor()
    print(f'Total: {time.time() - t0:.1f}s')
