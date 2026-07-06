
`timescale 1ns/1ps
`default_nettype none

module frame_ctrl_fsm #(
    parameter int ADDR_W = 11,
    parameter int OUT_W  = 24
) (
    input  wire                    clk,
    input  wire                    rstn,
    // UART RX
    input  wire [7:0]              rx_data,
    input  wire                    rx_valid,
    // UART TX
    output reg                     tx_start,
    output wire [7:0]              tx_data,
    input  wire                    tx_busy,
    // core load ports
    output wire                    q_we,
    output wire [ADDR_W-1:0]       q_waddr,
    output wire signed [7:0]       q_wdata,
    output wire                    k_we,
    output wire [ADDR_W-1:0]       k_waddr,
    output wire signed [7:0]       k_wdata,
    output wire                    v_we,
    output wire [ADDR_W-1:0]       v_waddr,
    output wire signed [7:0]       v_wdata,
    // core control / readback
    output reg                     core_start,
    input  wire                    core_done,
    output wire [ADDR_W-1:0]       o_raddr,
    input  wire signed [OUT_W-1:0] o_rdata,
    // status
    output wire [2:0]              dbg_state
);
    `include "ifa7_pkg.svh"

    localparam int NEL  = IFA7_N * IFA7_DK;      // elements per matrix (2048)
    localparam int TOT  = 3 * NEL;               // total load bytes (6144)
    localparam int RCW  = $clog2(TOT + 1);       // 13

    typedef enum logic [2:0] {
        FC_LOAD, FC_START, FC_RUN, FC_FETCH, FC_TXREQ, FC_TXON, FC_TXOFF, FC_IDLE
    } st_t;
    st_t st;

    reg [RCW-1:0]    rxcnt;     // load byte counter
    reg [ADDR_W-1:0] eaddr;     // output element address
    reg [1:0]        bidx;      // output byte index (0=MSB..2=LSB)
    reg [OUT_W-1:0]  o_word;    // latched output element

    assign dbg_state = st;

    // ---- load decode (combinational write strobes) -----------------------
    wire loading = (st == FC_LOAD) & rx_valid;
    wire in_q = (rxcnt < RCW'(NEL));
    wire in_k = (rxcnt >= RCW'(NEL))   & (rxcnt < RCW'(2*NEL));
    wire in_v = (rxcnt >= RCW'(2*NEL)) & (rxcnt < RCW'(TOT));

    assign q_we    = loading & in_q;
    assign k_we    = loading & in_k;
    assign v_we    = loading & in_v;
    assign q_waddr = ADDR_W'(rxcnt);
    assign k_waddr = ADDR_W'(rxcnt - RCW'(NEL));
    assign v_waddr = ADDR_W'(rxcnt - RCW'(2*NEL));
    assign q_wdata = rx_data;
    assign k_wdata = rx_data;
    assign v_wdata = rx_data;

    assign o_raddr = eaddr;

    // ---- output byte select (big-endian) ---------------------------------
    assign tx_data = (bidx == 2'd0) ? o_word[OUT_W-1   -: 8] :
                     (bidx == 2'd1) ? o_word[OUT_W-9   -: 8] :
                                      o_word[7:0];

    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            st <= FC_LOAD; rxcnt <= '0; eaddr <= '0; bidx <= '0;
            o_word <= '0; tx_start <= 1'b0; core_start <= 1'b0;
        end else begin
            tx_start   <= 1'b0;
            core_start <= 1'b0;
            case (st)
                FC_LOAD: if (rx_valid) begin
                    if (rxcnt == RCW'(TOT-1)) st <= FC_START;
                    rxcnt <= rxcnt + 1'b1;
                end
                FC_START: begin core_start <= 1'b1; st <= FC_RUN; end
                FC_RUN:   if (core_done) begin eaddr <= '0; st <= FC_FETCH; end
                FC_FETCH: begin o_word <= o_rdata; bidx <= 2'd0; st <= FC_TXREQ; end
                FC_TXREQ: if (!tx_busy) begin tx_start <= 1'b1; st <= FC_TXON; end
                FC_TXON:  if (tx_busy)  st <= FC_TXOFF;     // transmission started
                FC_TXOFF: if (!tx_busy) begin              // byte complete
                    if (bidx == 2'd2) begin
                        if (eaddr == ADDR_W'(NEL-1)) st <= FC_IDLE;
                        else begin eaddr <= eaddr + 1'b1; st <= FC_FETCH; end
                    end else begin
                        bidx <= bidx + 1'b1; st <= FC_TXREQ;
                    end
                end
                // Re-runnable: after streaming a full result, rewind and wait
                // for the next frame (so the host can call repeatedly without a
                // board reset). core_done stays high until the next FC_START.
                FC_IDLE: begin rxcnt <= '0; eaddr <= '0; st <= FC_LOAD; end
                default: st <= FC_LOAD;
            endcase
        end
    end
endmodule

`default_nettype wire
