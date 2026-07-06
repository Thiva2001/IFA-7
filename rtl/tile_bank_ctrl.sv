// ============================================================================
// tile_bank_ctrl.sv  --  IFA-7 BRAM port-aware ping-pong tile bank
// ----------------------------------------------------------------------------
// Realises the proposal's contribution #3: a banking + scheduling discipline
// that removes dual-port BRAM contention between the matmul (producer) and the
// softmax (consumer) on a few-BRAM fabric.
//
// Two physical dual-port BRAM banks form a ping-pong pair.  In any tile phase:
//   * the producer (QK^T MAC) WRITES the current "fill" bank through port A;
//   * the consumer (online softmax) READS the previous "drain" bank via port B.
// Because producer and consumer always touch DIFFERENT banks, and within a bank
// writes use port A while reads use port B, the MAC and the softmax NEVER share
// a port in the same cycle -- the contention that collapses utilisation on
// large arrays (SystolicAttention) is eliminated by construction.
//
// `swap` toggles the phase at each tile boundary (driven by the scheduler).
// This is a reusable IP block; it is exercised by tb/tb_tile_bank_ctrl.sv.
// ============================================================================
`timescale 1ns/1ps
`default_nettype none

module tile_bank_ctrl #(
    parameter int DW    = 32,
    parameter int DEPTH = 256
) (
    input  wire            clk,
    input  wire            swap,                 // toggle ping/pong at tile edge
    // producer port A (write into the current fill bank)
    input  wire            wr_en,
    input  wire [$clog2(DEPTH)-1:0] wr_addr,
    input  wire [DW-1:0]   wr_data,
    // consumer port B (read from the current drain bank, 1-cycle latency)
    input  wire [$clog2(DEPTH)-1:0] rd_addr,
    output wire [DW-1:0]   rd_data,
    output wire            phase_o               // current fill-bank index
);
    (* ram_style = "block" *) reg [DW-1:0] bank0 [0:DEPTH-1];
    (* ram_style = "block" *) reg [DW-1:0] bank1 [0:DEPTH-1];

    reg phase;                                   // 0: fill bank0/drain bank1
    always_ff @(posedge clk) if (swap) phase <= ~phase;
    assign phase_o = phase;

    wire wsel = phase;                           // write (fill) bank
    wire rsel = ~phase;                          // read  (drain) bank

    // Port A : producer writes the fill bank
    always_ff @(posedge clk) begin
        if (wr_en &&  wsel == 1'b0) bank0[wr_addr] <= wr_data;
        if (wr_en &&  wsel == 1'b1) bank1[wr_addr] <= wr_data;
    end

    // Port B : consumer reads the drain bank (registered read -> BRAM inference)
    reg [DW-1:0] rd_reg;
    always_ff @(posedge clk) begin
        rd_reg <= (rsel == 1'b0) ? bank0[rd_addr] : bank1[rd_addr];
    end
    assign rd_data = rd_reg;
endmodule

`default_nettype wire
