"""
ifa7_config.py
==============
Single source of truth for every numeric / structural parameter of the IFA-7
attention engine.  Both the Python golden model AND the SystemVerilog RTL are
generated from (or kept consistent with) the constants in this file.

IFA-7 = Integer FlashAttention for Artix-7 (bare-metal XC7A100T / Nexys A7).

All arithmetic in the engine is INTEGER / FIXED-POINT.  The choices below are
made so that the streaming ("online softmax") FlashAttention recursion is

    * non-amplifying  (alpha = exp(m_old - m_new) in (0,1]  ->  contraction), and
    * bit-exactly reproducible in hardware (no floating point anywhere).

Fixed-point conventions
-----------------------
  Q/K/V            : signed INT8                       (DATA_W bits)
  S_raw = Q.K^T    : signed integer                    (ACC_W bits)
  S (scaled score) : signed, SF fractional bits        -> Q(.,SF)
  p = exp(.)       : unsigned, PF fractional bits, 1.0 == (1<<PF)
  ell (row denom)  : unsigned, PF fractional bits       (L_W bits)
  acc (PV accum)   : signed,   PF fractional bits       (ACCO_W bits)
  O (output)       : signed,   OUT_F fractional bits    (OUT_W bits)

This module is plain Python (only the standard library + math) so it can run
anywhere; numpy is only used by the golden model / vector generator.
"""

import math

# --------------------------------------------------------------------------
# 1. Problem shape (verification defaults; all are RTL parameters / generics)
# --------------------------------------------------------------------------
# These small-but-representative sizes keep simulation fast while exercising
# every datapath.  They map cleanly onto a ViT-Tiny head (DK=64, scale N) by
# changing only these numbers -- see docs/Architecture.md.
DATA_W = 8     # Q/K/V element width (signed INT8)
DK     = 32    # head dimension d_k
N      = 64    # sequence length N (number of queries == number of keys)
BR     = 16    # query-row tile height  (FlashAttention B_r)
BC     = 16    # key/value column block (FlashAttention B_c)

assert N % BR == 0, "N must be a multiple of BR"
assert N % BC == 0, "N must be a multiple of BC"
NBLK = N // BC   # number of key/value blocks streamed per query row

# --------------------------------------------------------------------------
# 2. Score path  (S = (Q.K^T) * (1/sqrt(DK))  in fixed point Q(.,SF))
# --------------------------------------------------------------------------
ACC_W    = 32          # QK^T accumulator width (signed)
SF       = 8           # fractional bits of the scaled score S (chosen by sweep)
SCALE_SH = 12          # score scale: S = (S_raw * SCALE_M) >>> SCALE_SH

# SCALE_M chosen so that  SCALE_M / 2^SCALE_SH ~= (1/sqrt(DK)) * 2^SF
_score_factor = (1.0 / math.sqrt(DK)) * (1 << SF)
SCALE_M = int(round(_score_factor * (1 << SCALE_SH)))   # multiplier constant

S_W = ACC_W            # width carried for scaled score (re-use ACC_W, signed)

# --------------------------------------------------------------------------
# 3. exp() unit   exp(x) = 2^(x * log2e),  x <= 0  (argument is S - m, m_o - m)
# --------------------------------------------------------------------------
#   g  = x * LOG2E_FIX                       (G fractional bits added)
#   TF = SF + G   -> total fractional bits of the base-2 exponent
#   gn = -g >= 0 ; ishift = gn >> TF ; frac = gn & ((1<<TF)-1)
#   idx = frac >> (TF - LUT_BITS)            (top LUT_BITS of the fraction)
#   p   = EXP_LUT[idx] >> ishift             (clamp to 0 if ishift > PF)
PF       = 16          # probability fractional bits ; exp(0) == (1<<PF)
G        = 14          # extra fractional bits from the log2(e) multiply
LOG2E_FIX = int(round(math.log2(math.e) * (1 << G)))   # log2(e) in Q(.,G)
TF       = SF + G      # total fractional bits of the exponent
LUT_BITS = 8           # exp LUT address bits  -> LUT_DEPTH entries
LUT_DEPTH = 1 << LUT_BITS
LUT_W    = PF + 2      # LUT data width (value range (0, 1<<PF])

assert TF >= LUT_BITS, "TF must be >= LUT_BITS"

# Clamp: when the (positive) base-2 exponent magnitude exceeds PF the result is
# guaranteed < 1 LSB, so p == 0.  This is the proposal's "tail clamp" and it
# also bounds the hardware shifter to [0, PF].
P_W       = PF + 1     # width of a probability value (max == 1<<PF)
P_ZERO_GN = (PF + 1) << TF   # gn >= this  ->  p = 0

# --------------------------------------------------------------------------
# 4. Online-softmax accumulators  (ell, acc) and final divide
# --------------------------------------------------------------------------
M_INIT   = -(1 << 28)  # running-max sentinel (-inf surrogate, fits in S_W)
L_W      = 40          # running denominator ell width (unsigned)
ACCO_W   = 48          # PV output accumulator width (signed)
OUT_F    = 8           # output fractional bits
OUT_W    = 24          # output element width (signed)

# --------------------------------------------------------------------------
# 5. Clocking + UART framing (PL-only stream front-end)
# --------------------------------------------------------------------------
# The Nexys A7 oscillator is 100 MHz, but the fused arithmetic datapath (the
# combinational exp + un-pipelined DSP chains) does NOT close timing at 100 MHz
# on the -1 part (measured WNS ~= -13 ns). The whole design is therefore clocked
# on-chip at CLK_IN_HZ / CORE_DIV (25 MHz), which closes comfortably and leaves
# the already-verified RTL logic byte-for-byte unchanged. nexys_a7_top.sv
# generates this divided clock; CORE_DIV here MUST match the divider in top.
CLK_IN_HZ   = 100_000_000   # physical Nexys A7 oscillator (pin E3)
CORE_DIV    = 4             # on-chip clock divide -> 25 MHz core clock
CORE_CLK_HZ = CLK_IN_HZ // CORE_DIV
BAUD        = 115200
UART_DIV    = CORE_CLK_HZ // BAUD   # baud divider referenced to the CORE clock
CLK_HZ      = CORE_CLK_HZ   # (kept for backward compatibility)

# --------------------------------------------------------------------------
# Helper: exp LUT contents (shared by Python and RTL via a .mem file)
# --------------------------------------------------------------------------
def build_exp_lut():
    """EXP_LUT[i] = round( 2^(-i / LUT_DEPTH) * 2^PF ), i in [0, LUT_DEPTH)."""
    lut = []
    for i in range(LUT_DEPTH):
        val = round(math.pow(2.0, -float(i) / LUT_DEPTH) * (1 << PF))
        # i==0 -> 2^PF (== 1.0).  Clamp into LUT_W just in case.
        val = min(val, (1 << PF))
        lut.append(val)
    return lut


def emit_exp_mem(path):
    """Write the exp LUT as Verilog $readmemh hex, one entry per line."""
    lut = build_exp_lut()
    hexdigits = (LUT_W + 3) // 4
    with open(path, "w") as f:
        f.write(f"// IFA-7 exp LUT : EXP_LUT[i] = round(2^(-i/{LUT_DEPTH}) * 2^{PF})\n")
        f.write(f"// depth={LUT_DEPTH} width={LUT_W} bits (auto-generated, do not edit)\n")
        for v in lut:
            f.write(f"{v:0{hexdigits}x}\n")
    return lut


def emit_exp_init(path):
    """Emit rtl/exp_lut_init.svh: inline ROM initialisation (no file I/O at
    elaboration), guaranteeing identical contents in simulation and synthesis."""
    lut = build_exp_lut()
    with open(path, "w") as f:
        f.write("// AUTO-GENERATED exp ROM init (include inside an initial block)\n")
        f.write(f"// EXP_LUT[i] = round(2^(-i/{LUT_DEPTH}) * 2^{PF}), depth {LUT_DEPTH}\n")
        for i, v in enumerate(lut):
            f.write(f"rom[{i}] = {LUT_W}'h{v:0{(LUT_W + 3) // 4}x};\n")
    return lut


def emit_verilog_pkg(path):
    """Emit ifa7_pkg.svh with all localparams kept in lock-step with Python."""
    L = []
    L.append("// ----------------------------------------------------------------")
    L.append("// ifa7_pkg.svh  --  AUTO-GENERATED by python/ifa7_config.py")
    L.append("// Do not edit by hand: regenerate with  `py gen_vectors.py`.")
    L.append("// Single source of truth for every IFA-7 numeric/structural param.")
    L.append("//")
    L.append("// NOTE: NO include guard -- this file is `include-d inside each")
    L.append("// module body to create module-scoped localparams.  A guard would")
    L.append("// let only the first including module see the parameters.")
    L.append("// ----------------------------------------------------------------")
    items = [
        ("DATA_W", DATA_W), ("DK", DK), ("N", N), ("BR", BR), ("BC", BC),
        ("NBLK", NBLK), ("ACC_W", ACC_W), ("SF", SF), ("SCALE_SH", SCALE_SH),
        ("SCALE_M", SCALE_M), ("S_W", S_W), ("PF", PF), ("G", G),
        ("LOG2E_FIX", LOG2E_FIX), ("TF", TF), ("LUT_BITS", LUT_BITS),
        ("LUT_DEPTH", LUT_DEPTH), ("LUT_W", LUT_W), ("P_W", P_W),
        ("P_ZERO_GN", P_ZERO_GN), ("M_INIT", M_INIT), ("L_W", L_W),
        ("ACCO_W", ACCO_W), ("OUT_F", OUT_F), ("OUT_W", OUT_W),
        ("CORE_DIV", CORE_DIV), ("UART_DIV", UART_DIV),
    ]
    # Emit as signed `int` so the negative M_INIT sentinel is represented
    # correctly; all values fit within 32-bit signed range.
    for name, val in items:
        L.append(f"localparam int IFA7_{name} = {val};")
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    # Quick self-print of the derived constants (sanity check for humans).
    print("IFA-7 configuration")
    print(f"  DATA_W={DATA_W} DK={DK} N={N} BR={BR} BC={BC} NBLK={NBLK}")
    print(f"  SF={SF} SCALE_M={SCALE_M} SCALE_SH={SCALE_SH} "
          f"(factor~={SCALE_M/(1<<SCALE_SH):.5f}, ideal={_score_factor:.5f})")
    print(f"  PF={PF} G={G} LOG2E_FIX={LOG2E_FIX} TF={TF} "
          f"LUT_BITS={LUT_BITS} LUT_DEPTH={LUT_DEPTH} LUT_W={LUT_W}")
    print(f"  P_ZERO_GN={P_ZERO_GN} M_INIT={M_INIT}")
    print(f"  L_W={L_W} ACCO_W={ACCO_W} OUT_F={OUT_F} OUT_W={OUT_W}")
    print(f"  CORE_DIV={CORE_DIV} CORE_CLK_HZ={CORE_CLK_HZ} "
          f"UART_DIV={UART_DIV} (in={CLK_IN_HZ} BAUD={BAUD})")
    lut = build_exp_lut()
    print(f"  EXP_LUT[0]={lut[0]} (==1.0) EXP_LUT[{LUT_DEPTH//2}]={lut[LUT_DEPTH//2]} "
          f"(~={lut[LUT_DEPTH//2]/(1<<PF):.5f}, ideal={2**-0.5:.5f}) "
          f"EXP_LUT[{LUT_DEPTH-1}]={lut[LUT_DEPTH-1]}")
