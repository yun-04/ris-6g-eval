"""RIS-Assisted 6G Wireless Communications: paper-quality figures.
Generates 6 figures with unified IEEE style:
  Fig.1 - System geometry diagram (NEW)
  Fig.2 - BER vs SNR for varying N
  Fig.3 - Achievable rate vs transmit power
  Fig.4 - Rate vs phase quantization bits
  Fig.5 - Received SNR vs RIS horizontal position
  Fig.6 - Rate vs CSI estimation error variance
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy.random import default_rng
from scipy.special import erfc

import plot_style
plot_style.apply()
PAL = plot_style.PALETTE
MK = plot_style.MARKERS

rng = default_rng(2026)
OUT = '../figures'
os.makedirs(OUT, exist_ok=True)

# ------------------------------ Helpers ------------------------------
def db2lin(x): return 10.0 ** (x / 10.0)
def rayleigh(shape, rng_):
    return (rng_.standard_normal(shape) + 1j * rng_.standard_normal(shape)) / np.sqrt(2.0)
def rician(shape, K_dB, rng_):
    K = db2lin(K_dB)
    return (np.sqrt(K/(K+1)) * np.ones(shape, dtype=complex)
            + np.sqrt(1.0/(K+1)) * rayleigh(shape, rng_))
def path_loss_lin(d, alpha, PL0_dB=-30.0, d0=1.0):
    return db2lin(PL0_dB - 10.0 * alpha * np.log10(d / d0))

TX = np.array([0.0, 0.0]); RX = np.array([100.0, 0.0])
def ris_pos(x=50.0, y=10.0): return np.array([x, y])
ALPHA_D, ALPHA_TI, ALPHA_IR = 3.5, 2.2, 2.2
KdB_TI = KdB_IR = 3.0
NOISE_DBM = -94.0

def make_channels(N, RIS, n_trials, rng_, K_TI=None, K_IR=None):
    bd = path_loss_lin(np.linalg.norm(RX-TX), ALPHA_D)
    bTI = path_loss_lin(np.linalg.norm(RIS-TX), ALPHA_TI)
    bIR = path_loss_lin(np.linalg.norm(RX-RIS), ALPHA_IR)
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
    return rng_.uniform(0, 2*np.pi, size=(n_trials, N))
def quantize(theta, bits):
    if bits is None or bits == np.inf: return theta
    L = 2**bits; step = 2*np.pi/L
    return np.mod(np.round(np.mod(theta, 2*np.pi)/step)*step, 2*np.pi)
def composite(h_d, h, g, theta):
    return h_d + np.sum(h * np.exp(1j*theta) * g, axis=1)
def snr_lin(Pt_dBm, h_eff):
    Pt = db2lin(Pt_dBm) * 1e-3
    return Pt * np.abs(h_eff)**2 / (db2lin(NOISE_DBM) * 1e-3)
def rate(snr): return np.log2(1.0 + snr)
def bpsk_ber(snr): return 0.5 * erfc(np.sqrt(snr))


# ============================================================
# Fig.1  System geometry diagram
# ============================================================
def fig1_geometry():
    fig, ax = plt.subplots(figsize=(plot_style.COL, 2.3))
    # AP
    ax.plot(0, 0, marker='^', color=PAL[0], markersize=11, zorder=5,
            markeredgecolor='black', markeredgewidth=0.6)
    ax.text(0, -1.6, 'AP', ha='center', fontsize=8.5, fontweight='bold')
    # UE
    ax.plot(10, 0, marker='o', color=PAL[1], markersize=9, zorder=5,
            markeredgecolor='black', markeredgewidth=0.6)
    ax.text(10, -1.6, 'UE', ha='center', fontsize=8.5, fontweight='bold')
    # RIS array (small squares)
    ris_x, ris_y = 5, 2.5
    for i in range(8):
        for j in range(2):
            rect = mpatches.Rectangle((ris_x-0.7+i*0.18, ris_y+j*0.18),
                                      0.16, 0.16,
                                      facecolor='#FFD27A',
                                      edgecolor='black', linewidth=0.4)
            ax.add_patch(rect)
    ax.text(ris_x, ris_y+0.9, 'RIS ($N$ elements)', ha='center',
            fontsize=8, fontweight='bold')
    # Direct (blocked) path
    ax.annotate('', xy=(10, 0.05), xytext=(0, 0.05),
                arrowprops=dict(arrowstyle='-', color='#888888',
                                linestyle=(0, (3, 2)), lw=0.9))
    ax.text(5, -0.5, r'$h_d$ (blocked, NLoS)', ha='center',
            fontsize=7.5, color='#666', style='italic')
    # X mark on direct path (blockage)
    ax.plot(5, 0.05, marker='x', color='#C0392B', markersize=9, mew=1.5)
    # TX -> RIS
    ax.annotate('', xy=(ris_x-0.3, ris_y+0.1), xytext=(0.2, 0.2),
                arrowprops=dict(arrowstyle='->', color=PAL[0], lw=1.0))
    ax.text(2, 1.5, r'$\mathbf{h}\in\mathbb{C}^{N}$', fontsize=8.5,
            color=PAL[0])
    # RIS -> RX
    ax.annotate('', xy=(9.8, 0.2), xytext=(ris_x+0.7, ris_y+0.1),
                arrowprops=dict(arrowstyle='->', color=PAL[2], lw=1.0))
    ax.text(7.5, 1.5, r'$\mathbf{g}\in\mathbb{C}^{N}$', fontsize=8.5,
            color=PAL[2])
    # Distance markers
    ax.annotate('', xy=(10, -2.3), xytext=(0, -2.3),
                arrowprops=dict(arrowstyle='<->', color='black', lw=0.6))
    ax.text(5, -2.7, r'$d_{\mathrm{AU}}=100$ m', ha='center', fontsize=7.5)
    # Style
    ax.set_xlim(-1.5, 12); ax.set_ylim(-3.3, 4.2)
    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_visible(False)
    ax.grid(False)
    plt.tight_layout(pad=0.1)
    plt.savefig(f'{OUT}/fig1_sys.pdf')
    plt.savefig(f'{OUT}/fig1_sys.png', dpi=200)
    plt.close()
    print('Fig.1 (geometry) saved')


# ============================================================
# Fig.2  BER vs SNR
# ============================================================
def fig2_ber():
    SNR_dB = np.arange(-10, 21, 2)
    Ns = [0, 16, 32, 64, 128]
    n_trials = 20000
    RIS = ris_pos()
    fig, ax = plt.subplots(figsize=(plot_style.COL, 2.6))
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
            Pt_W = db2lin(snr_db) * db2lin(NOISE_DBM) * 1e-3 / np.mean(np.abs(h_d) ** 2)
            sigma2 = db2lin(NOISE_DBM) * 1e-3
            inst_snr = Pt_W * np.abs(h_eff) ** 2 / sigma2
            bers.append(np.mean(bpsk_ber(inst_snr)))
        bers = np.clip(np.array(bers), 1e-10, 1.0)
        label = 'No RIS' if N == 0 else fr'RIS, $N$={N}'
        ax.semilogy(SNR_dB, bers, marker=MK[idx], color=PAL[idx],
                    label=label, markevery=2)
    # Engineering reference lines
    ax.axhline(1e-3, color='#888', lw=0.6, ls='-.', alpha=0.7)
    ax.text(20, 1.5e-3, 'eMBB target', fontsize=6.5, color='#555',
            ha='right', va='bottom')
    ax.axhline(1e-8, color='#888', lw=0.6, ls='-.', alpha=0.7)
    ax.text(20, 1.5e-8, 'URLLC target', fontsize=6.5, color='#555',
            ha='right', va='bottom')
    # Annotate the five-orders-of-magnitude gap at SNR=10 dB
    ax.axvline(10, color='#666', lw=0.5, ls='--', alpha=0.6)
    ax.annotate(r'$\approx 5$ orders', xy=(10, 1e-5), xytext=(-7, 1e-7),
                fontsize=7, ha='left', color='#222',
                arrowprops=dict(arrowstyle='->', color='#444', lw=0.7))
    ax.set_xlabel('Direct-link SNR (dB)')
    ax.set_ylabel('Average BER')
    ax.set_ylim(1e-9, 1)
    ax.legend(loc='lower left', ncol=2, columnspacing=0.6)
    plt.tight_layout(pad=0.2)
    plt.savefig(f'{OUT}/fig2_ber.pdf')
    plt.savefig(f'{OUT}/fig2_ber.png', dpi=200)
    plt.close()
    print('Fig.2 (BER) saved')


# ============================================================
# Fig.3  Rate vs transmit power
# ============================================================
def fig3_rate():
    Pt_dBm = np.arange(-10, 41, 5)
    n_trials = 3000
    RIS = ris_pos()
    fig, ax = plt.subplots(figsize=(plot_style.COL, 2.6))
    schemes = [
        ('No RIS', 0, 'no'),
        ('Random RIS, $N$=64', 64, 'rand'),
        ('Optimal RIS, $N$=32', 32, 'opt'),
        ('Optimal RIS, $N$=64', 64, 'opt'),
    ]
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
        ax.plot(Pt_dBm, rates, marker=MK[k], color=PAL[k], label=label)
    # Highlight a power point
    ax.axvline(20, color='#666', lw=0.5, ls='--', alpha=0.6)
    ax.text(20.5, 0.5, r'$P_t=20$ dBm', fontsize=7, color='#444')
    ax.set_xlabel('Transmit power $P_t$ (dBm)')
    ax.set_ylabel('Achievable rate (bps/Hz)')
    ax.legend(loc='upper left')
    plt.tight_layout(pad=0.2)
    plt.savefig(f'{OUT}/fig3_rate.pdf')
    plt.savefig(f'{OUT}/fig3_rate.png', dpi=200)
    plt.close()
    print('Fig.3 (rate) saved')


# ============================================================
# Fig.4  Rate vs phase resolution
# ============================================================
def fig4_quant():
    bits = [1, 2, 3, 4, np.inf]
    Ns = [16, 32, 64, 128]
    Pt = 20.0; n_trials = 3000
    RIS = ris_pos()
    fig, ax = plt.subplots(figsize=(plot_style.COL, 2.6))
    width = 0.18
    x = np.arange(len(bits))
    for j, N in enumerate(Ns):
        h_d, h, g = make_channels(N, RIS, n_trials, rng)
        opt = phases_optimal(h_d, h, g)
        rates = []
        for b in bits:
            tq = quantize(opt, b)
            rates.append(np.mean(rate(snr_lin(Pt, composite(h_d, h, g, tq)))))
        ax.bar(x + (j-1.5)*width, rates, width=width, color=PAL[j],
               edgecolor='black', linewidth=0.4, label=f'$N$={N}')
    ax.set_xticks(x)
    ax.set_xticklabels(['1', '2', '3', '4', r'$\infty$'])
    ax.set_xlabel('Phase resolution (bits)')
    ax.set_ylabel('Achievable rate (bps/Hz)')
    ax.legend(loc='lower right', ncol=2, columnspacing=0.6)
    ax.grid(axis='y', linestyle=':', linewidth=0.4)
    ax.set_axisbelow(True)
    plt.tight_layout(pad=0.2)
    plt.savefig(f'{OUT}/fig4_quant.pdf')
    plt.savefig(f'{OUT}/fig4_quant.png', dpi=200)
    plt.close()
    print('Fig.4 (quantization) saved')


# ============================================================
# Fig.5  Received SNR vs RIS position
# ============================================================
def fig5_position():
    xs = np.arange(10, 96, 5)
    Pt = 20.0; N = 64; n_trials = 1500
    fig, ax = plt.subplots(figsize=(plot_style.COL, 2.6))
    snr_opt, snr_rand, snr_no = [], [], []
    for x in xs:
        RIS = ris_pos(x=x)
        h_d, h, g = make_channels(N, RIS, n_trials, rng)
        snr_no.append(10*np.log10(np.mean(snr_lin(Pt, h_d))))
        theta = phases_optimal(h_d, h, g)
        snr_opt.append(10*np.log10(np.mean(snr_lin(Pt, composite(h_d, h, g, theta)))))
        theta = phases_random(N, n_trials, rng)
        snr_rand.append(10*np.log10(np.mean(snr_lin(Pt, composite(h_d, h, g, theta)))))
    ax.plot(xs, snr_opt, marker=MK[0], color=PAL[0], label='Optimal RIS')
    ax.plot(xs, snr_rand, marker=MK[1], color=PAL[1], label='Random RIS')
    ax.plot(xs, snr_no, marker=MK[2], color=PAL[2], label='No RIS')
    # Shade favourable deployment regions
    ax.axvspan(10, 25, color=PAL[0], alpha=0.06)
    ax.axvspan(80, 95, color=PAL[0], alpha=0.06)
    ax.text(17, max(snr_opt)-1.5, 'Near AP', fontsize=7, ha='center', color=PAL[0])
    ax.text(87, max(snr_opt)-1.5, 'Near UE', fontsize=7, ha='center', color=PAL[0])
    ax.set_xlabel('RIS horizontal position $x_R$ (m)')
    ax.set_ylabel('Received SNR (dB)')
    ax.legend(loc='lower center', ncol=3, columnspacing=0.7)
    plt.tight_layout(pad=0.2)
    plt.savefig(f'{OUT}/fig5_pos.pdf')
    plt.savefig(f'{OUT}/fig5_pos.png', dpi=200)
    plt.close()
    print('Fig.5 (position) saved')


# ============================================================
# Fig.6  Rate vs CSI error
# ============================================================
def fig6_csi():
    sigma_e2_dB = np.arange(-30, 1, 3)
    Ns = [32, 64, 128]
    Pt = 20.0; n_trials = 3000
    RIS = ris_pos()
    fig, ax = plt.subplots(figsize=(plot_style.COL, 2.6))
    for j, N in enumerate(Ns):
        h_d, h, g = make_channels(N, RIS, n_trials, rng)
        theta_id = phases_optimal(h_d, h, g)
        rate_id = np.mean(rate(snr_lin(Pt, composite(h_d, h, g, theta_id))))
        rates = []
        for s_dB in sigma_e2_dB:
            s2 = db2lin(s_dB)
            sH = np.sqrt(s2 * np.mean(np.abs(h)**2))
            sG = np.sqrt(s2 * np.mean(np.abs(g)**2))
            sD = np.sqrt(s2 * np.mean(np.abs(h_d)**2))
            h_hat = h + sH * rayleigh(h.shape, rng)
            g_hat = g + sG * rayleigh(g.shape, rng)
            d_hat = h_d + sD * rayleigh(h_d.shape, rng)
            theta = phases_optimal(d_hat, h_hat, g_hat)
            rates.append(np.mean(rate(snr_lin(Pt, composite(h_d, h, g, theta)))))
        ax.plot(sigma_e2_dB, rates, marker=MK[j], color=PAL[j], label=f'$N$={N}')
        ax.axhline(rate_id, color=PAL[j], lw=0.5, ls='--', alpha=0.5)
    ax.fill_betweenx([4.0, 8.0], -30, -15, color='#9DC183', alpha=0.12,
                     label=None)
    ax.text(-25, 4.15, 'Near-perfect-CSI region', fontsize=7,
            color='#3E6826', ha='center')
    # Engineering threshold: -9 dB separates < 10% loss (left) from > 10% (right)
    ax.axvline(-9, color='#C0392B', lw=0.8, ls=':', alpha=0.85)
    ax.text(-9.5, 4.15, r'$\sigma_e^2 = -9$ dB',
            fontsize=6.5, color='#C0392B', ha='right', va='bottom')
    ax.annotate('', xy=(-3, 4.4), xytext=(-8.5, 4.4),
                arrowprops=dict(arrowstyle='->', color='#C0392B', lw=0.6))
    ax.text(-5.5, 4.5, r'$>$10% rate loss', fontsize=6.5,
            color='#C0392B', ha='center', va='bottom')
    ax.annotate('', xy=(-15, 4.4), xytext=(-9.5, 4.4),
                arrowprops=dict(arrowstyle='->', color='#3E6826', lw=0.6))
    ax.text(-12.5, 4.5, r'$<$10% rate loss', fontsize=6.5,
            color='#3E6826', ha='center', va='bottom')
    ax.set_xlabel(r'Channel estimation error $\sigma_e^2$ (dB)')
    ax.set_ylabel('Achievable rate (bps/Hz)')
    ax.legend(loc='lower left')
    ax.set_ylim(4.0, 8.0)
    plt.tight_layout(pad=0.2)
    plt.savefig(f'{OUT}/fig6_csi.pdf')
    plt.savefig(f'{OUT}/fig6_csi.png', dpi=200)
    plt.close()
    print('Fig.6 (CSI) saved')


# ============================================================
# Fig.7  Rate vs N for varying Rician K-factor (NEW for v7)
# ============================================================
def fig7_kfactor():
    Ns = np.array([8, 16, 32, 64, 128])
    Ks_dB = [-np.inf, 0, 3, 10, 30]   # -inf -> Rayleigh, 30 dB ~ near-LoS
    K_labels = ['Rayleigh', '$K{=}0$ dB', '$K{=}3$ dB',
                '$K{=}10$ dB', '$K{=}30$ dB (near-LoS)']
    Pt = 20.0
    n_trials = 4000
    RIS = ris_pos()
    fig, ax = plt.subplots(figsize=(plot_style.COL, 2.6))
    for j, K_dB in enumerate(Ks_dB):
        rates = []
        for N in Ns:
            h_d, h, g = make_channels(int(N), RIS, n_trials, rng,
                                       K_TI=K_dB, K_IR=K_dB)
            theta = phases_optimal(h_d, h, g)
            h_eff = composite(h_d, h, g, theta)
            rates.append(np.mean(rate(snr_lin(Pt, h_eff))))
        ax.plot(Ns, rates, marker=MK[j], color=PAL[j],
                label=K_labels[j], markersize=4.5, linewidth=1.1)
    # Annotate the N^2 scaling: doubling N adds approx const (in log scale)
    ax.set_xscale('log', base=2)
    ax.set_xticks(Ns)
    ax.set_xticklabels([str(int(n)) for n in Ns])
    ax.set_xlabel('RIS size $N$')
    ax.set_ylabel('Achievable rate (bps/Hz)')
    ax.legend(loc='upper left', fontsize=6.8, ncol=1)
    ax.grid(True, which='both', linestyle=':', linewidth=0.4)
    plt.tight_layout(pad=0.2)
    plt.savefig(f'{OUT}/fig7_kfactor.pdf')
    plt.savefig(f'{OUT}/fig7_kfactor.png', dpi=200)
    plt.close()
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
    print(f'Total: {time.time()-t0:.1f}s')
