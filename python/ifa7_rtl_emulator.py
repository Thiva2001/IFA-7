
import numpy as np

import ifa7_config as C
import ifa7_fixedpoint as FX

# FSM states (mirrors the typedef enum in the RTL)
(S_IDLE, S_RINIT, S_BINIT, S_SMAC, S_SSTORE, S_MNEW, S_PROB, S_ELL,
 S_PVK, S_PVMAC, S_PVSTORE, S_BNEXT, S_NISSUE, S_NWAIT, S_RNEXT, S_DONE) = range(16)

DIV_DW = C.ACCO_W + C.OUT_F


def _exp(x):
    """RTL exp_unit semantics: clamp x>0 to 0, else exp_fixed."""
    return FX.exp_fixed(x if x <= 0 else 0)


class CoreEmu:
    def __init__(self, Q, K, V):
        self.q = Q.flatten().astype(np.int64)
        self.k = K.flatten().astype(np.int64)
        self.v = V.flatten().astype(np.int64)
        self.o = np.zeros(C.N * C.DK, dtype=np.int64)
        # registers
        self.state = S_IDLE
        self.done = 0
        self.ii = self.bb = self.jj = self.kk = 0
        self.m = 0
        self.blkmax = 0
        self.ell = 0
        self.alpha = 0
        self.sum_p = 0
        self.macc = 0
        self.pvacc = 0
        self.s_arr = [0] * C.BC
        self.p_arr = [0] * C.BC
        self.acc = [0] * C.DK
        
        self.div_busy = 0
        self.div_cnt = 0
        self.div_q = 0
        self.div_done = 0
        self.div_dividend = 0
        self.div_divisor = 0

    
    def q_idx(self):  return self.ii * C.DK + self.kk
    def kv_row(self): return self.bb * C.BC + self.jj
    def kk_idx(self): return self.kv_row() * C.DK + self.kk
    def o_idx(self):  return self.ii * C.DK + self.kk

    def step_divider(self):
        self.div_done = 0
        div_start = (self.state == S_NISSUE)
        if div_start and not self.div_busy:
            acc_k = self.acc[self.kk]
            self.div_divisor = self.ell
            self.div_dividend = abs(acc_k) << C.OUT_F
            if self.div_divisor == 0:
                self.div_q = 0
                self.div_done = 1          # immediate (matches RTL guard)
                self.div_busy = 0
            else:
                self.div_busy = 1
                self.div_cnt = DIV_DW
        elif self.div_busy:
            self.div_cnt -= 1
            if self.div_cnt == 0:
                self.div_q = self.div_dividend // self.div_divisor
                self.div_busy = 0
                self.div_done = 1

    
    def step(self):
        st = self.state
        # snapshot combinational values
        s_scaled = FX.score_scale(self.macc)
        m_new_w = self.blkmax if self.blkmax > self.m else self.m

        if st == S_IDLE:
            pass  # waits for start (driven externally)
        elif st == S_RINIT:
            self.acc[self.kk] = 0
            if self.kk == C.DK - 1:
                self.m = C.M_INIT
                self.ell = 0
                self.bb = 0
                self.state = S_BINIT
            else:
                self.kk += 1
        elif st == S_BINIT:
            self.blkmax = C.M_INIT
            self.jj = 0
            self.kk = 0
            self.macc = 0
            self.state = S_SMAC
        elif st == S_SMAC:
            self.macc = self.macc + int(self.q[self.q_idx()]) * int(self.k[self.kk_idx()])
            if self.kk == C.DK - 1:
                self.state = S_SSTORE
            else:
                self.kk += 1
        elif st == S_SSTORE:
            self.s_arr[self.jj] = s_scaled
            if s_scaled > self.blkmax:
                self.blkmax = s_scaled
            if self.jj == C.BC - 1:
                self.state = S_MNEW
            else:
                self.jj += 1
                self.kk = 0
                self.macc = 0
                self.state = S_SMAC
        elif st == S_MNEW:
            self.alpha = _exp(self.m - m_new_w)     # exp(m_old - m_new)
            self.m = m_new_w
            self.jj = 0
            self.sum_p = 0
            self.state = S_PROB
        elif st == S_PROB:
            p = _exp(self.s_arr[self.jj] - self.m)   # exp(S - m_new)
            self.p_arr[self.jj] = p
            self.sum_p = self.sum_p + p
            if self.jj == C.BC - 1:
                self.state = S_ELL
            else:
                self.jj += 1
        elif st == S_ELL:
            self.ell = FX.rescale_mul(self.alpha, self.ell) + self.sum_p
            self.kk = 0
            self.state = S_PVK
        elif st == S_PVK:
            self.jj = 0
            self.pvacc = 0
            self.state = S_PVMAC
        elif st == S_PVMAC:
            self.pvacc = self.pvacc + self.p_arr[self.jj] * int(self.v[self.kk_idx()])
            if self.jj == C.BC - 1:
                self.state = S_PVSTORE
            else:
                self.jj += 1
        elif st == S_PVSTORE:
            self.acc[self.kk] = FX.rescale_mul(self.alpha, self.acc[self.kk]) + self.pvacc
            if self.kk == C.DK - 1:
                self.state = S_BNEXT
            else:
                self.kk += 1
                self.state = S_PVK
        elif st == S_BNEXT:
            if self.bb == C.NBLK - 1:
                self.kk = 0
                self.state = S_NISSUE
            else:
                self.bb += 1
                self.state = S_BINIT
        elif st == S_NISSUE:
            self.state = S_NWAIT
        elif st == S_NWAIT:
            if self.div_done:
                acc_k = self.acc[self.kk]
                mag = self.div_q
                self.o[self.o_idx()] = -mag if acc_k < 0 else mag
                if self.kk == C.DK - 1:
                    self.state = S_RNEXT
                else:
                    self.kk += 1
                    self.state = S_NISSUE
        elif st == S_RNEXT:
            if self.ii == C.N - 1:
                self.state = S_DONE
            else:
                self.ii += 1
                self.kk = 0
                self.state = S_RINIT
        elif st == S_DONE:
            self.done = 1
            self.state = S_IDLE

    def run(self, max_cycles=5_000_000):
        # start pulse
        self.state = S_RINIT
        self.ii = 0
        self.kk = 0
        cycles = 0
        while cycles < max_cycles:
            self.step_divider()
            self.step()
            cycles += 1
            if self.done:
                break
        if not self.done:
            raise RuntimeError("emulator did not finish (deadlock?)")
        return self.o.reshape(C.N, C.DK), cycles


def emulate(Q, K, V):
    emu = CoreEmu(Q, K, V)
    return emu.run()


if __name__ == "__main__":
    import ifa7_golden as G
    rs = np.random.RandomState(0xA7)
    Q = rs.randint(-6, 7, (C.N, C.DK)).astype(np.int64)
    K = rs.randint(-6, 7, (C.N, C.DK)).astype(np.int64)
    V = rs.randint(-128, 128, (C.N, C.DK)).astype(np.int64)
    O_emu, ncyc = emulate(Q, K, V)
    O_gold = G.attention_fixed(Q, K, V)
    ok = np.array_equal(O_emu, O_gold)
    print(f"emulator cycles = {ncyc}")
    print(f"RTL-FSM emulator == golden attention_fixed : {ok}")
    if not ok:
        diff = np.argwhere(O_emu != O_gold)
        print("first mismatches:", diff[:5].tolist())
        for (i, k) in diff[:5]:
            print(f"  [{i},{k}] emu={O_emu[i,k]} gold={O_gold[i,k]}")
