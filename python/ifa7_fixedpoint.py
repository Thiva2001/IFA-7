
import ifa7_config as C

EXP_LUT = C.build_exp_lut()


_ROUND_F = 1 << (C.TF - C.LUT_BITS - 1)       # half-LSB at the LUT fraction step


def exp_fixed(x):
    """Fixed-point exp() for x <= 0 (argument already in Q(.,SF)).

    Returns p in Q(.,PF), i.e. exp(0) -> (1<<PF).  Implements
        exp(x) = 2^(x*log2e) = round( EXP_LUT[idx] / 2^ishift )   (clamped to 0)
    where the (idx, ishift) split is itself round-to-nearest with carry from the
    fraction into ishift.  This is exactly rtl/exp_unit.sv.
    """
    assert x <= 0, "exp_fixed only valid for x <= 0 (online-softmax invariant)"
    g  = x * C.LOG2E_FIX          # <= 0 ; Q(., SF+G) == Q(., TF)
    gn = -g                       # >= 0
    if gn >= C.P_ZERO_GN:         # tail clamp -> probability underflows to 0
        return 0
    gnr    = gn + _ROUND_F                     # round to LUT fraction resolution
    ishift = gnr >> C.TF                        # integer #halvings (0..PF)
    if ishift > C.PF:                           # bounded clamp (paranoia)
        return 0
    idx    = (gnr >> (C.TF - C.LUT_BITS)) & (C.LUT_DEPTH - 1)
    base   = EXP_LUT[idx]
    if ishift == 0:
        return base
    return (base + (1 << (ishift - 1))) >> ishift   # round-to-nearest shift


def score_scale(s_raw):
    """Scale raw QK^T by 1/sqrt(DK): S = (s_raw * SCALE_M) >>> SCALE_SH.

    Uses arithmetic (floor) shift so it matches Verilog `>>>` on a signed value.
    """
    return (s_raw * C.SCALE_M) >> C.SCALE_SH


def rescale_mul(alpha, value):
    """(alpha * value) >>> PF  with alpha in Q(.,PF), value signed or unsigned.

    Python floor-shift matches Verilog arithmetic shift for both signs.
    """
    return (alpha * value) >> C.PF


def divide_trunc(num, den):
    """Truncate-toward-zero division (sign-magnitude), matches rtl/divider.sv."""
    if den == 0:
        return 0
    sign = -1 if (num < 0) else 1
    q = abs(num) // den          # floor of magnitude == truncation toward zero
    return sign * q


def wrap_signed(value, width):
    """Two's-complement wrap into `width` bits (for overflow-behaviour checks)."""
    mask = (1 << width) - 1
    value &= mask
    if value & (1 << (width - 1)):
        value -= (1 << width)
    return value
