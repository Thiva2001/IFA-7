// ============================================================================
// ifa7_attention_core.sv  --  IFA-7 fully-integer online-softmax attention core
// ----------------------------------------------------------------------------
// Single-head scaled-dot-product attention with FlashAttention-style streaming
// (online softmax).  This module is the bit-exact hardware twin of
// python/ifa7_golden.py:attention_fixed().
//
//   For each query row i:
//     m = -inf ; ell = 0 ; acc[*] = 0
//     for each key/value block b (BC columns):
//        S^(b)   = scale( Q_i . K_j )          (QK^T MAC + 1/sqrt(d))
//        m_new   = max(m, rowmax(S^(b)))
//        alpha   = exp(m - m_new)              (in (0,1]  -> contraction)
//        p       = exp(S^(b) - m_new)          (clamped tail)
//        ell     = alpha*ell + sum(p)
//        acc[k]  = alpha*acc[k] + sum_j p_j*V[j,k]    (PV MAC)
//     O[i,k] = (acc[k] << OUT_F) / ell          (delayed division)
//
// Numerics (single fixed-point truth): see python/ifa7_config.py + ifa7_pkg.svh.
//
// Datapath style: this is the *correctness-reference* microarchitecture -- a
// sequential MAC (1 multiply/cycle) feeding a fused online-softmax/PV pipeline.
// It is fully synthesizable and verifies bit-exactly against the golden model.
// The throughput-optimised variant (DSP-packed BR x BC array, port-aware
// ping-pong banks via tile_bank_ctrl.sv) is described in docs/Architecture.md;
// its resource/latency figures are the proposal's analytical projections.
// ============================================================================
`timescale 1ns/1ps
`default_nettype none

module ifa7_attention_core #(
    // These defaults MUST equal $clog2(IFA7_N*IFA7_DK) and IFA7_OUT_W; a
    // generate-time assertion below enforces it (parameters are needed in the
    // header because the package include happens in the body).
    parameter int ADDR_W = 11,
    parameter int OUT_W  = 24
) (
    input  wire                       clk,
    input  wire                       rstn,
    input  wire                       start,    // pulse: begin one attention pass
    output reg                        done,     // level: high when O is ready

    // ---- load ports (element addressed, row-major idx = row*DK + col) ----
    input  wire                       q_we,
    input  wire [ADDR_W-1:0]          q_waddr,
    input  wire signed [7:0]          q_wdata,
    input  wire                       k_we,
    input  wire [ADDR_W-1:0]          k_waddr,
    input  wire signed [7:0]          k_wdata,
    input  wire                       v_we,
    input  wire [ADDR_W-1:0]          v_waddr,
    input  wire signed [7:0]          v_wdata,

    // ---- output read port (combinational) ----
    input  wire [ADDR_W-1:0]          o_raddr,
    output wire signed [OUT_W-1:0]    o_rdata
);
    `include "ifa7_pkg.svh"

    // ---- local shorthands -------------------------------------------------
    localparam int N      = IFA7_N;
    localparam int DK     = IFA7_DK;
    localparam int BC     = IFA7_BC;
    localparam int NBLK   = IFA7_NBLK;
    localparam int DATA_W = IFA7_DATA_W;
    localparam int S_W    = IFA7_S_W;
    localparam int P_W    = IFA7_P_W;
    localparam int PF     = IFA7_PF;
    localparam int L_W    = IFA7_L_W;
    localparam int ACCO_W = IFA7_ACCO_W;
    localparam int ACC_W  = IFA7_ACC_W;
    localparam int OUT_F  = IFA7_OUT_F;

    // Check that the header parameters match the package truth (sim-time).
    initial begin
        if (ADDR_W != $clog2(IFA7_N * IFA7_DK))
            $error("ifa7_attention_core: ADDR_W must be %0d", $clog2(IFA7_N*IFA7_DK));
        if (OUT_W != IFA7_OUT_W)
            $error("ifa7_attention_core: OUT_W must be %0d", IFA7_OUT_W);
    end

    localparam int DIV_DW = ACCO_W + OUT_F;             // 56
    localparam int II_W   = $clog2(N);                  // 6
    localparam int BB_W   = (NBLK > 1) ? $clog2(NBLK) : 1;
    localparam int JJ_W   = $clog2(BC);                 // 4
    localparam int KK_W   = $clog2(DK);                 // 5

    // ---- memories (row-major; combinational read keeps the MAC hazard-free)
    (* ram_style = "distributed" *) reg signed [DATA_W-1:0] q_mem [0:N*DK-1];
    (* ram_style = "distributed" *) reg signed [DATA_W-1:0] k_mem [0:N*DK-1];
    (* ram_style = "distributed" *) reg signed [DATA_W-1:0] v_mem [0:N*DK-1];
    (* ram_style = "distributed" *) reg signed [OUT_W-1:0]  o_mem [0:N*DK-1];

    always_ff @(posedge clk) begin
        if (q_we) q_mem[q_waddr] <= q_wdata;
        if (k_we) k_mem[k_waddr] <= k_wdata;
        if (v_we) v_mem[v_waddr] <= v_wdata;
    end
    assign o_rdata = o_mem[o_raddr];

    // ---- working storage --------------------------------------------------
    reg signed [S_W-1:0]    s_arr  [0:BC-1];
    reg        [P_W-1:0]    p_arr  [0:BC-1];
    reg signed [ACCO_W-1:0] acc_arr[0:DK-1];

    reg signed [S_W-1:0]    m_reg;        // running max (m_old, then m_new)
    reg signed [S_W-1:0]    blkmax;       // rowmax of current block
    reg        [L_W-1:0]    ell;          // running denominator
    reg        [P_W-1:0]    alpha;        // exp(m_old - m_new)
    reg        [L_W-1:0]    sum_p;        // sum of block probabilities
    reg signed [ACC_W-1:0]  macc;         // QK^T accumulator
    reg signed [ACCO_W-1:0] pvacc;        // PV accumulator

    reg [II_W-1:0] ii;
    reg [BB_W-1:0] bb;
    reg [JJ_W-1:0] jj;
    reg [KK_W-1:0] kk;

    // ---- index arithmetic -------------------------------------------------
    wire [ADDR_W-1:0] kv_row = bb*BC + jj;             // j0 + jj
    wire [ADDR_W-1:0] q_idx  = ii*DK + kk;
    wire [ADDR_W-1:0] kk_idx = kv_row*DK + kk;
    wire [ADDR_W-1:0] o_idx  = ii*DK + kk;

    // ---- score scaling: S = (macc * SCALE_M) >>> SCALE_SH -----------------
    wire signed [ACC_W+20-1:0] sprod    = macc * $signed(20'(IFA7_SCALE_M));
    wire signed [S_W-1:0]      s_scaled = S_W'(sprod >>> IFA7_SCALE_SH);

    // ---- m_new = max(m_old, blkmax) ; exp() input mux ---------------------
    wire signed [S_W-1:0] m_new_w = (blkmax > m_reg) ? blkmax : m_reg;
    reg  signed [S_W-1:0] exp_x;
    wire        [P_W-1:0] exp_p;

    // ---- alpha*ell and alpha*acc rescale (>>> PF) -------------------------
    wire [P_W+L_W-1:0]    ellp     = alpha * ell;
    wire [L_W-1:0]        ell_resc = L_W'(ellp >> PF);
    // product of an 18-bit signed (0,alpha) and a 48-bit signed -> 66 bits
    wire signed [P_W+ACCO_W:0]   accp = $signed({1'b0, alpha}) * acc_arr[kk];
    wire signed [ACCO_W-1:0]     acc_resc = ACCO_W'(accp >>> PF);

    // ---- PV partial product  p_arr[jj] * V[j,k] (signed) ------------------
    wire signed [P_W+DATA_W:0] pv_prod =
            $signed({1'b0, p_arr[jj]}) * $signed(v_mem[kk_idx]);

    // ---- FSM states -------------------------------------------------------
    typedef enum logic [4:0] {
        S_IDLE, S_RINIT, S_BINIT, S_SMAC, S_SSTORE, S_MNEW, S_PROB, S_ELL,
        S_PVK, S_PVMAC, S_PVSTORE, S_BNEXT, S_NISSUE, S_NWAIT, S_RNEXT, S_DONE
    } state_t;
    state_t state;

    // ---- final-divide datapath -------------------------------------------
    wire signed [ACCO_W-1:0] acc_k   = acc_arr[kk];
    wire                     o_sign  = acc_k[ACCO_W-1];
    wire        [ACCO_W-1:0] acc_abs = o_sign ? (~acc_k + 1'b1) : acc_k;
    wire        [DIV_DW-1:0] dividend = {acc_abs, {OUT_F{1'b0}}};
    wire                     div_start = (state == S_NISSUE);
    wire                     div_busy, div_done;
    wire        [DIV_DW-1:0] div_q;

    divider #(.DW(DIV_DW), .VW(L_W)) u_div (
        .clk(clk), .rstn(rstn), .start(div_start),
        .dividend(dividend), .divisor(ell),
        .busy(div_busy), .done(div_done),
        .quotient(div_q), .remainder() );

    wire signed [OUT_W-1:0] o_mag    = $signed({1'b0, div_q[OUT_W-2:0]});
    wire signed [OUT_W-1:0] o_signed = o_sign ? -o_mag : o_mag;

    exp_unit u_exp (.x(exp_x), .p(exp_p));   // S_W==32, P_W==17 (see exp_unit)

    // ---- exp() argument mux (combinational) -------------------------------
    always_comb begin
        if (state == S_MNEW) exp_x = m_reg - m_new_w;   // m_old - m_new  (<=0)
        else                 exp_x = s_arr[jj] - m_reg; // S - m_new      (<=0)
    end

    // ---- main sequential FSM ---------------------------------------------
    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            state <= S_IDLE; done <= 1'b0;
            ii <= '0; bb <= '0; jj <= '0; kk <= '0;
            m_reg <= '0; blkmax <= '0; ell <= '0; alpha <= '0;
            sum_p <= '0; macc <= '0; pvacc <= '0;
        end else begin
            case (state)
            // ------------------------------------------------------------
            S_IDLE: begin
                if (start) begin
                    done <= 1'b0; ii <= '0; kk <= '0; state <= S_RINIT;
                end
            end
            // clear acc[*] then init row-level accumulators
            S_RINIT: begin
                acc_arr[kk] <= '0;
                if (kk == KK_W'(DK-1)) begin
                    m_reg <= S_W'(IFA7_M_INIT); ell <= '0;
                    bb <= '0; state <= S_BINIT;
                end else kk <= kk + 1'b1;
            end
            // begin a key/value block
            S_BINIT: begin
                blkmax <= S_W'(IFA7_M_INIT);
                jj <= '0; kk <= '0; macc <= '0; state <= S_SMAC;
            end
            // QK^T : macc += Q[i,kk]*K[j,kk]
            S_SMAC: begin
                macc <= macc + $signed(q_mem[q_idx]) * $signed(k_mem[kk_idx]);
                if (kk == KK_W'(DK-1)) state <= S_SSTORE;
                else kk <= kk + 1'b1;
            end
            // scale, store score, track block max
            S_SSTORE: begin
                s_arr[jj] <= s_scaled;
                if (s_scaled > blkmax) blkmax <= s_scaled;
                if (jj == JJ_W'(BC-1)) state <= S_MNEW;
                else begin jj <= jj + 1'b1; kk <= '0; macc <= '0; state <= S_SMAC; end
            end
            // m_new = max(m_old, blkmax) ; alpha = exp(m_old - m_new)
            S_MNEW: begin
                alpha <= exp_p;          // exp_x = m_old - m_new (mux)
                m_reg <= m_new_w;
                jj <= '0; sum_p <= '0; state <= S_PROB;
            end
            // p[jj] = exp(S[jj] - m_new) ; sum_p += p
            S_PROB: begin
                p_arr[jj] <= exp_p;      // exp_x = s_arr[jj] - m_new (mux)
                sum_p <= sum_p + L_W'(exp_p);
                if (jj == JJ_W'(BC-1)) state <= S_ELL;
                else jj <= jj + 1'b1;
            end
            // ell = alpha*ell + sum_p
            S_ELL: begin
                ell <= ell_resc + sum_p;
                kk <= '0; state <= S_PVK;
            end
            // begin PV accumulation for output dim kk
            S_PVK: begin
                jj <= '0; pvacc <= '0; state <= S_PVMAC;
            end
            // pvacc += p[jj]*V[j,kk]
            S_PVMAC: begin
                pvacc <= pvacc + ACCO_W'(pv_prod);
                if (jj == JJ_W'(BC-1)) state <= S_PVSTORE;
                else jj <= jj + 1'b1;
            end
            // acc[kk] = alpha*acc[kk] + pvacc
            S_PVSTORE: begin
                acc_arr[kk] <= acc_resc + pvacc;
                if (kk == KK_W'(DK-1)) state <= S_BNEXT;
                else begin kk <= kk + 1'b1; state <= S_PVK; end
            end
            // next block or normalise
            S_BNEXT: begin
                if (bb == BB_W'(NBLK-1)) begin kk <= '0; state <= S_NISSUE; end
                else begin bb <= bb + 1'b1; state <= S_BINIT; end
            end
            // issue divide for acc[kk] (div_start = combinational pulse)
            S_NISSUE: begin
                state <= S_NWAIT;
            end
            // wait for divide, write O[i,kk]
            S_NWAIT: begin
                if (div_done) begin
                    o_mem[o_idx] <= o_signed;
                    if (kk == KK_W'(DK-1)) state <= S_RNEXT;
                    else begin kk <= kk + 1'b1; state <= S_NISSUE; end
                end
            end
            // next row or finish
            S_RNEXT: begin
                if (ii == II_W'(N-1)) state <= S_DONE;
                else begin ii <= ii + 1'b1; kk <= '0; state <= S_RINIT; end
            end
            S_DONE: begin
                done <= 1'b1; state <= S_IDLE;
            end
            default: state <= S_IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
