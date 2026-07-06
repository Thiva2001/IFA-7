// ============================================================================
// exp_unit.sv  --  IFA-7 fully-integer exponential  exp(x) for x <= 0
// ----------------------------------------------------------------------------
//   exp(x) = 2^(x * log2e)
//          = round( EXP_LUT[idx] / 2^ishift )        (clamped to 0 in the tail)
//
// where the (idx, ishift) split is round-to-nearest with carry from the
// fraction into the integer shift.  This module is the bit-exact hardware twin
// of python/ifa7_fixedpoint.py:exp_fixed().  It is purely combinational; the
// caller registers the result.
//
// Input  x : signed score difference (S - m  or  m_old - m_new), always <= 0
//            by the online-softmax invariant (enforced/clamped here too).
// Output p : unsigned probability in Q(.,PF), exp(0) == (1<<PF).
//
// This is the heart of the proposal's "retraining-free fixed-point softmax":
// because the argument is always <= 0 the result is always in (0, 1<<PF], i.e.
// the recursion is non-amplifying (contraction) and cannot overflow.
// ============================================================================
`timescale 1ns/1ps
`default_nettype none

module exp_unit (
    input  wire signed [31:0] x,     // S_W-wide; <= 0 (clamped internally)
    output wire        [16:0] p      // P_W-wide; in [0, 1<<PF]
);
    `include "ifa7_pkg.svh"

    localparam int GW      = IFA7_S_W + 16;                 // product width
    localparam int TF      = IFA7_TF;                       // 22
    localparam int LUT_B   = IFA7_LUT_BITS;                 // 8
    localparam int LUT_W   = IFA7_LUT_W;                    // 18
    localparam int PF      = IFA7_PF;                       // 16
    localparam logic signed [15:0] LOG2E16 = IFA7_LOG2E_FIX;
    localparam logic [GW-1:0] PZ     = IFA7_P_ZERO_GN;   // zero-extends (fits 27b)
    localparam logic [GW-1:0] ROUNDF = (GW'(1) << (TF - LUT_B - 1)); // 1<<13

    // Port widths are hardcoded (32, 17) to match IFA7_S_W / IFA7_P_W.
    initial begin
        if (IFA7_S_W != 32) $error("exp_unit: expects IFA7_S_W==32");
        if (IFA7_P_W != 17) $error("exp_unit: expects IFA7_P_W==17");
    end

    // Enforce the x <= 0 invariant (if x>0 ever appears, treat as 0 -> exp=1).
    wire signed [IFA7_S_W-1:0] xc = (x > 0) ? '0 : x;

    // g = x * log2e  (<= 0) ;  gn = |g|
    wire signed [GW-1:0] g  = xc * LOG2E16;
    wire        [GW-1:0] gn = $unsigned(-g);

    // round to LUT fraction resolution, then split into shift + index
    wire        [GW-1:0] gnr        = gn + ROUNDF;
    wire        [GW-1:0] ishift_full = gnr >> TF;
    wire        [LUT_B-1:0] idx      = gnr[TF-1 -: LUT_B];   // bits [21:14]
    wire                 zero_tail   = (gn >= PZ);
    wire                 over        = (ishift_full > GW'(PF));
    wire        [4:0]    ishift      = ishift_full[4:0];     // <= 16 otherwise

    // fractional mantissa lookup
    wire [LUT_W-1:0] base;
    exp_lut #(.LUT_BITS(LUT_B), .LUT_W(LUT_W)) u_rom (.addr(idx), .data(base));

    // round-to-nearest arithmetic shift right by `ishift`
    wire [LUT_W-1:0] round_add = (ishift == 5'd0) ? '0
                                                  : (LUT_W'(1) << (ishift - 5'd1));
    wire [LUT_W:0]   sum     = base + round_add;
    wire [LUT_W:0]   shifted = sum >> ishift;

    assign p = (zero_tail || over) ? 17'd0 : shifted[16:0];
endmodule

`default_nettype wire
