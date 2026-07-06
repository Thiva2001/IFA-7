// ============================================================================
// tb_frame_core.sv  --  integration test: frame_ctrl_fsm + ifa7_attention_core
// ----------------------------------------------------------------------------
// Validates the on-board datapath WITHOUT the slow bit-level UART: it drives
// frame_ctrl's parallel byte interface (rx_data/rx_valid) with the Q/K/V byte
// stream, models a UART transmitter (captures tx_data on the tx_start/tx_busy
// handshake), then deserialises the captured 3-byte big-endian output elements
// and compares them to sim/vectors/o_expected.mem.
//
// This covers exactly what the LEDs showed failing on the board: load
// addressing, the run trigger, output streaming/byte-order, the TX handshake,
// and the re-run loopback. Run from sim/ (or via the Vivado project sim_1).
// ============================================================================
`timescale 1ns/1ps
`ifndef IFA7_VEC_DIR
  `define IFA7_VEC_DIR "vectors/"
`endif

module tb_frame_core;
    `include "ifa7_pkg.svh"

    localparam int ADDR_W = $clog2(IFA7_N * IFA7_DK);
    localparam int OUT_W  = IFA7_OUT_W;
    localparam int NEL    = IFA7_N * IFA7_DK;     // 2048
    localparam int TOTIN  = 3 * NEL;              // 6144 input bytes
    localparam int TOTOUT = 3 * NEL;              // 6144 output bytes

    logic clk = 0, rstn;
    always #5 clk = ~clk;

    // frame_ctrl <-> core nets
    logic [7:0] rx_data; logic rx_valid;
    logic       tx_start; logic [7:0] tx_data; logic tx_busy;
    logic       q_we, k_we, v_we;
    logic [ADDR_W-1:0] q_waddr, k_waddr, v_waddr, o_raddr;
    logic signed [7:0] q_wdata, k_wdata, v_wdata;
    logic       core_start, core_done;
    logic signed [OUT_W-1:0] o_rdata;
    logic [2:0] dbg_state;

    frame_ctrl_fsm #(.ADDR_W(ADDR_W), .OUT_W(OUT_W)) u_fc (
        .clk(clk), .rstn(rstn),
        .rx_data(rx_data), .rx_valid(rx_valid),
        .tx_start(tx_start), .tx_data(tx_data), .tx_busy(tx_busy),
        .q_we(q_we), .q_waddr(q_waddr), .q_wdata(q_wdata),
        .k_we(k_we), .k_waddr(k_waddr), .k_wdata(k_wdata),
        .v_we(v_we), .v_waddr(v_waddr), .v_wdata(v_wdata),
        .core_start(core_start), .core_done(core_done),
        .o_raddr(o_raddr), .o_rdata(o_rdata), .dbg_state(dbg_state) );

    ifa7_attention_core #(.ADDR_W(ADDR_W), .OUT_W(OUT_W)) u_core (
        .clk(clk), .rstn(rstn), .start(core_start), .done(core_done),
        .q_we(q_we), .q_waddr(q_waddr), .q_wdata(q_wdata),
        .k_we(k_we), .k_waddr(k_waddr), .k_wdata(k_wdata),
        .v_we(v_we), .v_waddr(v_waddr), .v_wdata(v_wdata),
        .o_raddr(o_raddr), .o_rdata(o_rdata) );

    // ---- mock UART transmitter: capture bytes on the real handshake -------
    logic [7:0]  out_bytes [0:TOTOUT-1];
    integer      outcnt;
    logic        mock_busy; logic [3:0] txc;
    assign tx_busy = mock_busy;
    always_ff @(posedge clk) begin
        if (!rstn) begin mock_busy <= 0; txc <= 0; outcnt <= 0; end
        else begin
            if (tx_start && !mock_busy) begin
                if (outcnt < TOTOUT) out_bytes[outcnt] <= tx_data;
                outcnt    <= outcnt + 1;
                mock_busy <= 1; txc <= 4;          // emulate a few busy cycles
            end else if (mock_busy) begin
                if (txc == 0) mock_busy <= 0; else txc <= txc - 1;
            end
        end
    end

    // ---- vectors ----------------------------------------------------------
    logic signed [7:0]       q_data [0:NEL-1];
    logic signed [7:0]       k_data [0:NEL-1];
    logic signed [7:0]       v_data [0:NEL-1];
    logic signed [OUT_W-1:0] o_exp  [0:NEL-1];
    logic [7:0]              in_bytes [0:TOTIN-1];

    integer i, errors;
    logic [OUT_W-1:0] got;

    initial begin
        #80_000_000; $display("TB_FRAME_CORE: TIMEOUT"); $finish;
    end

    initial begin
        $readmemh({`IFA7_VEC_DIR, "q.mem"},          q_data);
        $readmemh({`IFA7_VEC_DIR, "k.mem"},          k_data);
        $readmemh({`IFA7_VEC_DIR, "v.mem"},          v_data);
        $readmemh({`IFA7_VEC_DIR, "o_expected.mem"}, o_exp);
        for (i = 0; i < NEL; i = i + 1) begin
            in_bytes[i]         = q_data[i];
            in_bytes[NEL + i]   = k_data[i];
            in_bytes[2*NEL + i] = v_data[i];
        end

        rx_valid = 0; rx_data = 0;
        rstn = 0; repeat (5) @(negedge clk); rstn = 1; @(negedge clk);

        // feed the Q||K||V byte stream, one byte per cycle
        for (i = 0; i < TOTIN; i = i + 1) begin
            @(negedge clk); rx_valid = 1; rx_data = in_bytes[i];
        end
        @(negedge clk); rx_valid = 0;

        // wait until all output bytes have been captured
        wait (outcnt >= TOTOUT);
        @(negedge clk);

        // deserialise (3 bytes/element, big-endian) and compare
        errors = 0;
        for (i = 0; i < NEL; i = i + 1) begin
            got = {out_bytes[3*i], out_bytes[3*i+1], out_bytes[3*i+2]};
            if (got !== o_exp[i][OUT_W-1:0]) begin
                errors = errors + 1;
                if (errors <= 20)
                    $display("  MISMATCH elem %0d : got %h exp %h",
                             i, got, o_exp[i][OUT_W-1:0]);
            end
        end
        $display("tb_frame_core: %0d elements, %0d errors (core_done=%0b)",
                 NEL, errors, core_done);
        if (errors == 0) $display("TB_FRAME_CORE: PASS");
        else             $display("TB_FRAME_CORE: FAIL");
        $finish;
    end
endmodule
