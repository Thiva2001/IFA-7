
`timescale 1ns/1ps
`default_nettype none

module nexys_a7_top (
    input  wire        CLK100MHZ,
    input  wire        CPU_RESETN,     // active-low push button
    input  wire        UART_TXD_IN,    // host -> FPGA serial (FPGA receives)
    output wire        UART_RXD_OUT,   // FPGA -> host serial (FPGA transmits)
    output wire [15:0] LED
);
    `include "ifa7_pkg.svh"

    localparam int ADDR_W = $clog2(IFA7_N * IFA7_DK);
    localparam int OUT_W  = IFA7_OUT_W;

    // ---- on-chip clock divider: 100 MHz / 4 = 25 MHz ----------------------
    // NOTE: the divide ratio (here /4 via clkdiv[1]) MUST equal IFA7_CORE_DIV.
    // If timing still fails, increase the divide (e.g. clkdiv[2] for /8) and set
    // CORE_DIV accordingly in python/ifa7_config.py, then regenerate the pkg.
    (* keep = "true" *) reg [1:0] clkdiv = 2'b00;
    always_ff @(posedge CLK100MHZ) clkdiv <= clkdiv + 1'b1;

    wire clk;
    BUFG u_clkbuf (.I(clkdiv[1]), .O(clk));

    // ---- reset synchroniser (async assert, sync deassert) on core clock ---
    reg [1:0] rst_sync = 2'b00;
    always_ff @(posedge clk or negedge CPU_RESETN)
        if (!CPU_RESETN) rst_sync <= 2'b00;
        else             rst_sync <= {rst_sync[0], 1'b1};
    wire rstn = rst_sync[1];

    // ---- UART --------------------------------------------------------------
    wire [7:0] rx_data;  wire rx_valid;
    wire [7:0] tx_data;  wire tx_start, tx_busy;

    uart_rx #(.DIV(IFA7_UART_DIV)) u_rx (
        .clk(clk), .rstn(rstn), .rx(UART_TXD_IN),
        .data(rx_data), .valid(rx_valid) );

    uart_tx #(.DIV(IFA7_UART_DIV)) u_tx (
        .clk(clk), .rstn(rstn), .start(tx_start), .data(tx_data),
        .tx(UART_RXD_OUT), .busy(tx_busy) );

    // ---- frame controller <-> core ---------------------------------------
    wire                    q_we, k_we, v_we;
    wire [ADDR_W-1:0]       q_waddr, k_waddr, v_waddr;
    wire signed [7:0]       q_wdata, k_wdata, v_wdata;
    wire                    core_start, core_done;
    wire [ADDR_W-1:0]       o_raddr;
    wire signed [OUT_W-1:0] o_rdata;
    wire [2:0]              dbg_state;

    frame_ctrl_fsm #(.ADDR_W(ADDR_W), .OUT_W(OUT_W)) u_fc (
        .clk(clk), .rstn(rstn),
        .rx_data(rx_data), .rx_valid(rx_valid),
        .tx_start(tx_start), .tx_data(tx_data), .tx_busy(tx_busy),
        .q_we(q_we), .q_waddr(q_waddr), .q_wdata(q_wdata),
        .k_we(k_we), .k_waddr(k_waddr), .k_wdata(k_wdata),
        .v_we(v_we), .v_waddr(v_waddr), .v_wdata(v_wdata),
        .core_start(core_start), .core_done(core_done),
        .o_raddr(o_raddr), .o_rdata(o_rdata),
        .dbg_state(dbg_state) );

    ifa7_attention_core #(.ADDR_W(ADDR_W), .OUT_W(OUT_W)) u_core (
        .clk(clk), .rstn(rstn),
        .start(core_start), .done(core_done),
        .q_we(q_we), .q_waddr(q_waddr), .q_wdata(q_wdata),
        .k_we(k_we), .k_waddr(k_waddr), .k_wdata(k_wdata),
        .v_we(v_we), .v_waddr(v_waddr), .v_wdata(v_wdata),
        .o_raddr(o_raddr), .o_rdata(o_rdata) );

    // ---- heartbeat (proves the core clock is alive) -----------------------
    reg [24:0] hb = '0;
    always_ff @(posedge clk) hb <= hb + 1'b1;

    // ---- status LEDs ------------------------------------------------------
    assign LED[0]    = rstn;          // out of reset
    assign LED[1]    = core_done;     // a computation has completed
    assign LED[2]    = tx_busy;       // transmitting a result byte
    assign LED[5:3]  = dbg_state;     // frame_ctrl FSM state
    assign LED[14:6] = '0;
    assign LED[15]   = hb[24];        // ~0.75 Hz blink @25 MHz : clock alive
endmodule

`default_nettype wire
