"""
make_figures.py
===============
Generate the publication figures for IFA-7 into docs/figures/.
Reproducible: re-run `py make_figures.py` to regenerate.

Figures:
  fig_exp_approx.png      fixed-point exp() vs ideal, and its relative error
  fig_precision_sweep.png end-to-end RMS error vs score fractional bits SF
  fig_accuracy.png        fixed vs float output scatter + error histogram
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ifa7_config as C
import ifa7_fixedpoint as FX
import ifa7_golden as G

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(os.path.dirname(HERE), "docs", "figures")
os.makedirs(FIG, exist_ok=True)


def fig_exp_approx():
    xs = np.arange(0, -(20 << C.SF), -1)
    real = np.array([x / (1 << C.SF) for x in xs])
    fixed = np.array([FX.exp_fixed(int(x)) / (1 << C.PF) for x in xs])
    ideal = np.exp(real)
    rel = np.abs(fixed - ideal) / np.maximum(ideal, 1e-12)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(real, ideal, label="ideal $e^{x}$", lw=2)
    ax[0].plot(real, fixed, "--", label="IFA-7 fixed-point", lw=1.3)
    ax[0].set_xlabel("x"); ax[0].set_ylabel("exp(x)")
    ax[0].set_title("Fixed-point exp (LUT + shift)")
    ax[0].legend(); ax[0].grid(alpha=0.3)

    ax[1].semilogy(real, np.maximum(rel, 1e-6), color="C3", lw=1.2)
    ax[1].axhline(0.01, color="k", ls=":", label="1% bound")
    ax[1].set_xlabel("x"); ax[1].set_ylabel("relative error")
    ax[1].set_title("exp relative error")
    ax[1].legend(); ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_exp_approx.png"), dpi=140)
    plt.close(fig)


def fig_precision_sweep():
    import importlib
    SFs = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    rms = []
    for sf in SFs:
        C.SF = sf
        C._score_factor = (1.0 / math.sqrt(C.DK)) * (1 << sf)
        C.SCALE_M = int(round(C._score_factor * (1 << C.SCALE_SH)))
        C.TF = sf + C.G
        C.P_ZERO_GN = (C.PF + 1) << C.TF
        importlib.reload(FX); importlib.reload(G)
        errs = []
        for s in range(6):
            rs = np.random.RandomState(s)
            Q = rs.randint(-6, 7, (C.N, C.DK)).astype(np.int64)
            K = rs.randint(-6, 7, (C.N, C.DK)).astype(np.int64)
            V = rs.randint(-128, 128, (C.N, C.DK)).astype(np.int64)
            errs.append(G.error_stats(G.attention_fixed(Q, K, V),
                                      G.attention_float(Q, K, V))["rms"])
        rms.append(np.mean(errs))
    # restore SF=8
    C.SF = 8
    C._score_factor = (1.0 / math.sqrt(C.DK)) * (1 << 8)
    C.SCALE_M = int(round(C._score_factor * (1 << C.SCALE_SH)))
    C.TF = 8 + C.G; C.P_ZERO_GN = (C.PF + 1) << C.TF
    importlib.reload(FX); importlib.reload(G)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.semilogy(SFs, rms, "o-", lw=1.6)
    ax.axvline(8, color="C2", ls="--", label="chosen SF=8")
    ax.set_xlabel("score fractional bits  SF")
    ax.set_ylabel("end-to-end RMS error (V-units)")
    ax.set_title("Accuracy vs score precision (DK=%d, N=%d)" % (C.DK, C.N))
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_precision_sweep.png"), dpi=140)
    plt.close(fig)


def fig_accuracy():
    rs = np.random.RandomState(0xA7)
    Q = rs.randint(-6, 7, (C.N, C.DK)).astype(np.int64)
    K = rs.randint(-6, 7, (C.N, C.DK)).astype(np.int64)
    V = rs.randint(-128, 128, (C.N, C.DK)).astype(np.int64)
    Of = G.attention_float(Q, K, V)
    Ox = G.fixed_to_float(G.attention_fixed(Q, K, V))
    err = (Ox - Of).flatten()

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].scatter(Of.flatten(), Ox.flatten(), s=4, alpha=0.3)
    lim = [Of.min(), Of.max()]
    ax[0].plot(lim, lim, "k--", lw=1)
    ax[0].set_xlabel("float reference O"); ax[0].set_ylabel("IFA-7 fixed O")
    ax[0].set_title("Output correlation"); ax[0].grid(alpha=0.3)

    ax[1].hist(err, bins=60, color="C0")
    ax[1].set_xlabel("absolute error (V-units)"); ax[1].set_ylabel("count")
    ax[1].set_title("Error histogram (rms=%.3f)" % np.sqrt((err**2).mean()))
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_accuracy.png"), dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    fig_exp_approx()
    fig_precision_sweep()
    fig_accuracy()
    print("figures written to", FIG)
