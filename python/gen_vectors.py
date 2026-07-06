

import os
import numpy as np

import ifa7_config as C
import ifa7_fixedpoint as FX
import ifa7_golden as G

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RTL  = os.path.join(ROOT, "rtl")
VEC  = os.path.join(ROOT, "sim", "vectors")
os.makedirs(VEC, exist_ok=True)


def hexw(value, width):
    """Two's-complement hex string of `value` in `width` bits."""
    mask = (1 << width) - 1
    return f"{(value & mask):0{(width + 3) // 4}x}"


def write_mem(path, values, width, header=None):
    with open(path, "w") as f:
        if header:
            f.write(f"// {header}\n")
        for v in values:
            f.write(hexw(int(v), width) + "\n")


def main():
    seed = 0xA7
    rs = np.random.RandomState(seed)

    # ---- exp LUT + Verilog package (the shared single-source-of-truth) -----
    lut = C.emit_exp_mem(os.path.join(VEC, "exp_lut.mem"))
    C.emit_exp_init(os.path.join(RTL, "exp_lut_init.svh"))
    C.emit_verilog_pkg(os.path.join(RTL, "ifa7_pkg.svh"))

    # ---- directed exp(x) test pairs (covers 0, small, clamp tail) ----------
    xs = [0, -1, -16, -64, -256, -1024, -(1 << C.SF), -3 * (1 << C.SF),
          -(1 << (C.SF + 3)), -50000, -(C.P_ZERO_GN), -(C.P_ZERO_GN + 1)]
    # add a uniform sweep across the meaningful range
    xs += list(range(0, -(40 << C.SF), -(1 << (C.SF - 2)) or -1))
    pairs = [(x, FX.exp_fixed(x)) for x in xs]
    # No comment header: this file is consumed by $fscanf("%h %h") in the TB.
    with open(os.path.join(VEC, "exp_test.mem"), "w") as f:
        for x, p in pairs:
            f.write(f"{hexw(x, C.S_W)} {hexw(p, C.P_W)}\n")

    # ---- end-to-end attention vectors --------------------------------------
    # Modest Q/K range keeps the softmax non-degenerate (the hard, honest case).
    Q = rs.randint(-6, 7, size=(C.N, C.DK)).astype(np.int64)
    K = rs.randint(-6, 7, size=(C.N, C.DK)).astype(np.int64)
    V = rs.randint(-128, 128, size=(C.N, C.DK)).astype(np.int64)

    O_fixed = G.attention_fixed(Q, K, V)
    O_float = G.attention_float(Q, K, V)
    stats = G.error_stats(O_fixed, O_float)

    write_mem(os.path.join(VEC, "q.mem"), Q.flatten(), C.DATA_W, "Q row-major INT8")
    write_mem(os.path.join(VEC, "k.mem"), K.flatten(), C.DATA_W, "K row-major INT8")
    write_mem(os.path.join(VEC, "v.mem"), V.flatten(), C.DATA_W, "V row-major INT8")
    write_mem(os.path.join(VEC, "o_expected.mem"), O_fixed.flatten(), C.OUT_W,
              f"golden O  Q(.,{C.OUT_F})  OUT_W={C.OUT_W}")

    with open(os.path.join(VEC, "meta.txt"), "w") as f:
        f.write("IFA-7 simulation vectors\n")
        f.write(f"  seed              : 0x{seed:X}\n")
        f.write(f"  shape             : N={C.N} DK={C.DK} BC={C.BC} NBLK={C.NBLK}\n")
        f.write(f"  fixed-point       : SF={C.SF} PF={C.PF} TF={C.TF} "
                f"LUT_DEPTH={C.LUT_DEPTH} OUT_F={C.OUT_F}\n")
        f.write(f"  SCALE_M/SCALE_SH  : {C.SCALE_M}/{C.SCALE_SH}\n")
        f.write("  error (fixed vs float, output in real V-units):\n")
        for kk, vv in stats.items():
            f.write(f"      {kk:9s} = {vv:.6f}\n")

    print("Artifacts written to", VEC)
    print(f"  exp LUT depth {C.LUT_DEPTH}, exp_test vectors {len(pairs)}")
    print(f"  attention vectors: {C.N}x{C.DK} (N x DK)")
    print("  error (fixed vs float):", {k: round(v, 5) for k, v in stats.items()})
    return stats


if __name__ == "__main__":
    main()
