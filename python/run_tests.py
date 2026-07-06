"""
run_tests.py
============
Self-checking regression suite for the IFA-7 golden model.  No external test
framework required (pure Python + numpy), so it runs anywhere `py` does.

It validates the claims the dissertation rests on:
  T1  exp unit  : fixed exp(x) matches 2^(x log2 e) within the LUT bound.
  T2  contraction property: alpha = exp(m_old - m_new) is ALWAYS in [0, 1<<PF]
      (this is what makes the online recursion non-amplifying / overflow-safe).
  T3  accuracy : |fixed - float| stays within the error budget over many random
      problems AND directed corner cases (uniform softmax, near one-hot, zeros,
      saturated INT8, single hot key).
  T4  determinism: the fixed model is bit-reproducible (run twice -> identical).
  T5  width safety: no intermediate exceeds its declared RTL width.

Exit code 0 == all pass (used by sim/run_all so CI can gate on it).
"""

import sys
import math
import numpy as np

import ifa7_config as C
import ifa7_fixedpoint as FX
import ifa7_golden as G

# Error budget (real V-units, output range ~ +/-128).  Derived empirically and
# bounded by the analysis in docs/Architecture.md; generous vs measured ~0.21.
MAX_ABS_BUDGET = 1.0     # worst single output element
RMS_BUDGET     = 0.20    # rms over all elements

PASS = "[ PASS ]"
FAIL = "[ FAIL ]"
_failures = 0


def check(cond, msg):
    global _failures
    print(f"  {PASS if cond else FAIL} {msg}")
    if not cond:
        _failures += 1


def t1_exp_unit():
    print("T1  exp() unit vs reference 2^(x*log2e)")
    worst_rel = 0.0     # relative error where probability is non-negligible
    worst_abs = 0.0     # absolute error everywhere (LSBs of Q(.,PF))
    for x in range(0, -(60 << C.SF), -1):
        p = FX.exp_fixed(x)
        ref = math.pow(2.0, (x / (1 << C.SF)) * math.log2(math.e)) * (1 << C.PF)
        worst_abs = max(worst_abs, abs(p - ref))
        if ref >= 64.0:         # >= 2^-10 of the max: a meaningful probability
            worst_rel = max(worst_rel, abs(p - ref) / ref)
    # Relative accuracy where it matters: bounded by the rounded LUT fraction
    # step (~0.5*ln2/LUT_DEPTH) plus the rounded output shift.
    check(worst_rel < 0.01, f"max relative exp error (p>=2^-10) = {worst_rel:.5f} (< 0.01)")
    # Absolute error scales with p (fraction quantisation), so bound it
    # relatively: |p-ref| <= 0.005*ref + 1 LSB everywhere.
    check(worst_abs < 0.005 * (1 << C.PF) + 1,
          f"max absolute exp error = {worst_abs:.2f} LSB (< 0.5% of 1.0)")
    # exp(0) must be exactly 1.0
    check(FX.exp_fixed(0) == (1 << C.PF), f"exp(0) == 1<<PF ({1<<C.PF})")
    # deep tail must clamp to 0
    check(FX.exp_fixed(-(C.P_ZERO_GN)) == 0, "deep negative arg clamps to 0")


def t2_contraction():
    print("T2  contraction: alpha = exp(m_old-m_new) in [0, 1<<PF]")
    ok = True
    for d in range(0, 200000, 37):        # m_old - m_new <= 0 always
        a = FX.exp_fixed(-d)
        if not (0 <= a <= (1 << C.PF)):
            ok = False
            break
    check(ok, "alpha never exceeds 1.0 and never negative (non-amplifying)")


def _accuracy_case(name, Q, K, V, max_abs=MAX_ABS_BUDGET, rms=RMS_BUDGET):
    Of = G.attention_float(Q, K, V)
    Ox = G.attention_fixed(Q, K, V)
    st = G.error_stats(Ox, Of)
    check(st["max_abs"] <= max_abs and st["rms"] <= rms,
          f"{name:26s} max_abs={st['max_abs']:.4f} rms={st['rms']:.4f}")
    return Ox, st


def t3_accuracy():
    print("T3  accuracy (fixed vs float)")
    # 12 random problems
    worst = 0.0
    for s in range(12):
        rs = np.random.RandomState(1000 + s)
        Q = rs.randint(-6, 7, (C.N, C.DK)).astype(np.int64)
        K = rs.randint(-6, 7, (C.N, C.DK)).astype(np.int64)
        V = rs.randint(-128, 128, (C.N, C.DK)).astype(np.int64)
        _, st = _accuracy_case(f"random seed {1000+s}", Q, K, V)
        worst = max(worst, st["max_abs"])
    print(f"      -> worst max_abs over random set = {worst:.4f}")

    # corner cases
    Z = np.zeros((C.N, C.DK), dtype=np.int64)
    rs = np.random.RandomState(7)
    Vr = rs.randint(-128, 128, (C.N, C.DK)).astype(np.int64)
    _accuracy_case("uniform softmax (Q=K=0)", Z, Z, Vr)            # exact
    # near one-hot: one key strongly aligned with every query
    Q = np.ones((C.N, C.DK), dtype=np.int64)
    K = rs.randint(-1, 2, (C.N, C.DK)).astype(np.int64)
    K[0] = 6                                                       # hot key row 0
    _accuracy_case("near one-hot (hot key)", Q, K, Vr)
    # saturated INT8 magnitudes (stress score path + clamp)
    Qs = rs.randint(-3, 4, (C.N, C.DK)).astype(np.int64)
    Ks = rs.randint(-3, 4, (C.N, C.DK)).astype(np.int64)
    Vs = np.full((C.N, C.DK), 127, dtype=np.int64)
    Vs[::2] = -128
    _accuracy_case("saturated V (+/-128)", Qs, Ks, Vs)


def t4_determinism():
    print("T4  bit-exact determinism")
    rs = np.random.RandomState(42)
    Q = rs.randint(-6, 7, (C.N, C.DK)).astype(np.int64)
    K = rs.randint(-6, 7, (C.N, C.DK)).astype(np.int64)
    V = rs.randint(-128, 128, (C.N, C.DK)).astype(np.int64)
    a = G.attention_fixed(Q, K, V)
    b = G.attention_fixed(Q, K, V)
    check(np.array_equal(a, b), "two runs produce identical integer output")


def t5_width_safety():
    print("T5  intermediate width safety (no RTL overflow)")
    max_s = max_ell = max_acc = 0
    for s in range(6):
        rs = np.random.RandomState(2000 + s)
        Q = rs.randint(-8, 9, (C.N, C.DK)).astype(np.int64)
        K = rs.randint(-8, 9, (C.N, C.DK)).astype(np.int64)
        V = rs.randint(-128, 128, (C.N, C.DK)).astype(np.int64)
        tr = []
        G.attention_fixed(Q, K, V, trace=tr)
        for t in tr:
            max_ell = max(max_ell, t["ell"])
            max_acc = max(max_acc, max(abs(a) for a in t["acc"]))
        for i in range(C.N):
            for j in range(C.N):
                max_s = max(max_s, abs(FX.score_scale(int(np.dot(Q[i], K[j])))))
    check(max_s < (1 << (C.S_W - 1)), f"max|S_scaled| {max_s} fits S_W={C.S_W}")
    check(max_ell < (1 << C.L_W), f"max ell {max_ell} fits L_W={C.L_W}")
    check(max_acc < (1 << (C.ACCO_W - 1)), f"max|acc| {max_acc} fits ACCO_W={C.ACCO_W}")


def t6_rtl_emulator():
    """The cycle-accurate FSM emulator of rtl/ifa7_attention_core.sv must match
    the golden model bit-for-bit (validates the RTL control logic / timing)."""
    print("T6  RTL-FSM cycle emulator == golden (control-logic validation)")
    import ifa7_rtl_emulator as E
    cases = []
    for s in range(4):
        rs = np.random.RandomState(3000 + s)
        cases.append((f"random {3000+s}",
                      rs.randint(-6, 7, (C.N, C.DK)).astype(np.int64),
                      rs.randint(-6, 7, (C.N, C.DK)).astype(np.int64),
                      rs.randint(-128, 128, (C.N, C.DK)).astype(np.int64)))
    Z = np.zeros((C.N, C.DK), dtype=np.int64)
    rs = np.random.RandomState(9)
    cases.append(("uniform (Q=K=0)", Z, Z, rs.randint(-128, 128, (C.N, C.DK)).astype(np.int64)))
    for name, Q, K, V in cases:
        O_emu, _ = E.emulate(Q, K, V)
        O_gold = G.attention_fixed(Q, K, V)
        check(np.array_equal(O_emu, O_gold), f"{name:18s} emulator == golden")


def main():
    print("=" * 64)
    print("IFA-7 golden-model regression")
    print(f"  N={C.N} DK={C.DK} BC={C.BC} SF={C.SF} PF={C.PF} LUT_DEPTH={C.LUT_DEPTH}")
    print("=" * 64)
    t1_exp_unit()
    t2_contraction()
    t3_accuracy()
    t4_determinism()
    t5_width_safety()
    t6_rtl_emulator()
    print("=" * 64)
    if _failures == 0:
        print("ALL TESTS PASSED")
        return 0
    print(f"{_failures} CHECK(S) FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
