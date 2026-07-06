
import numpy as np

import ifa7_config as C
import ifa7_fixedpoint as FX


def attention_float(Q, K, V):
    Qf = Q.astype(np.float64)
    Kf = K.astype(np.float64)
    Vf = V.astype(np.float64)

    scale = C.SCALE_M / (1 << C.SCALE_SH) / (1 << C.SF)   # maps S_raw -> real S
    S = (Qf @ Kf.T) * scale                               # (N, N)
    S = S - S.max(axis=1, keepdims=True)
    P = np.exp(S)
    P = P / P.sum(axis=1, keepdims=True)
    O = P @ Vf                                             # (N, DK), real units
    return O


def attention_fixed(Q, K, V, trace=None):

    N, DK = Q.shape
    assert DK == C.DK and N == C.N
    Oout = np.zeros((N, DK), dtype=np.int64)

    for i in range(N):                       # one query row at a time
        m   = C.M_INIT                       # running max  (Q(.,SF) domain)
        ell = 0                              # running denominator (Q(.,PF))
        acc = [0] * DK                       # output accumulator  (Q(.,PF))

        for b in range(C.NBLK):              # stream key/value blocks
            j0 = b * C.BC
            
            s_blk = []
            for jj in range(C.BC):
                j = j0 + jj
                s_raw = int(np.dot(Q[i].astype(np.int64), K[j].astype(np.int64)))
                s_blk.append(FX.score_scale(s_raw))

            
            m_old = m
            m_new = m_old
            for s in s_blk:
                if s > m_new:
                    m_new = s

            
            alpha = FX.exp_fixed(m_old - m_new)   # arg <= 0

            
            p_blk = [FX.exp_fixed(s - m_new) for s in s_blk]   # each arg <= 0
            sum_p = sum(p_blk)

            
            ell = FX.rescale_mul(alpha, ell) + sum_p

            
            for k in range(DK):
                pv = 0
                for jj in range(C.BC):
                    pv += p_blk[jj] * int(V[j0 + jj, k])
                acc[k] = FX.rescale_mul(alpha, acc[k]) + pv

            if trace is not None:
                trace.append(dict(i=i, b=b, m=m_new, alpha=alpha,
                                  ell=ell, sum_p=sum_p, acc=list(acc)))
            m = m_new

        
        for k in range(DK):
            Oout[i, k] = FX.divide_trunc(acc[k] << C.OUT_F, ell)

    return Oout


def fixed_to_float(O_fixed):
    """Convert the integer output (Q(.,OUT_F)) back to real units for error calc."""
    return O_fixed.astype(np.float64) / (1 << C.OUT_F)


def error_stats(O_fixed, O_float):
    Of = fixed_to_float(O_fixed)
    abs_err = np.abs(Of - O_float)
    denom = np.maximum(np.abs(O_float), 1e-9)
    rel = abs_err / denom
    return dict(
        max_abs=float(abs_err.max()),
        mean_abs=float(abs_err.mean()),
        rms=float(np.sqrt((abs_err ** 2).mean())),
        max_rel=float(rel.max()),
        mean_rel=float(rel.mean()),
    )
