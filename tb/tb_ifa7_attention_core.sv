
`timescale 1ns/1ps
`ifndef IFA7_VEC_DIR
  `define IFA7_VEC_DIR "vectors/"
`endif

module tb_ifa7_attention_core;
    `include "ifa7_pkg.svh"

    localparam int ADDR_W = $clog2(IFA7_N * IFA7_DK);
    localparam int OUT_W  = IFA7_OUT_W;
    localparam int NEL    = IFA7_N * IFA7_DK;

    logic clk = 1'b0, rstn, start, done;
    logic q_we, k_we, v_we;
    logic [ADDR_W-1:0] q_waddr, k_waddr, v_waddr, o_raddr;
    logic signed [7:0] q_wdata, k_wdata, v_wdata;
    logic signed [OUT_W-1:0] o_rdata;

    ifa7_attention_core #(.ADDR_W(ADDR_W), .OUT_W(OUT_W)) dut (
        .clk(clk), .rstn(rstn), .start(start), .done(done),
        .q_we(q_we), .q_waddr(q_waddr), .q_wdata(q_wdata),
        .k_we(k_we), .k_waddr(k_waddr), .k_wdata(k_wdata),
        .v_we(v_we), .v_waddr(v_waddr), .v_wdata(v_wdata),
        .o_raddr(o_raddr), .o_rdata(o_rdata) );

    always #5 clk = ~clk;                              // 100 MHz

    // vector storage
    logic signed [7:0]       q_data [0:NEL-1];
    logic signed [7:0]       k_data [0:NEL-1];
    logic signed [7:0]       v_data [0:NEL-1];
    logic signed [OUT_W-1:0] o_exp  [0:NEL-1];

    integer idx, errors;

    // watchdog (sequential reference core ~0.41 M cycles ~ 4.1 ms @100 MHz)
    initial begin
        #50_000_000;                                  // 50 ms
        $display("TB_CORE: TIMEOUT (done never asserted)");
        $finish;
    end

    initial begin
        $readmemh({`IFA7_VEC_DIR, "q.mem"},          q_data);
        $readmemh({`IFA7_VEC_DIR, "k.mem"},          k_data);
        $readmemh({`IFA7_VEC_DIR, "v.mem"},          v_data);
        $readmemh({`IFA7_VEC_DIR, "o_expected.mem"}, o_exp);

        q_we = 0; k_we = 0; v_we = 0; start = 0;
        q_waddr = 0; k_waddr = 0; v_waddr = 0; o_raddr = 0;
        q_wdata = 0; k_wdata = 0; v_wdata = 0;

        // reset
        rstn = 0;
        repeat (5) @(negedge clk);
        rstn = 1;
        @(negedge clk);

        // load Q, K, V (one element/cycle, addresses 0..NEL-1)
        for (idx = 0; idx < NEL; idx = idx + 1) begin
            @(negedge clk);
            q_we = 1; q_waddr = idx[ADDR_W-1:0]; q_wdata = q_data[idx];
            k_we = 1; k_waddr = idx[ADDR_W-1:0]; k_wdata = k_data[idx];
            v_we = 1; v_waddr = idx[ADDR_W-1:0]; v_wdata = v_data[idx];
        end
        @(negedge clk);
        q_we = 0; k_we = 0; v_we = 0;

        // start the attention pass
        @(negedge clk); start = 1;
        @(negedge clk); start = 0;

        // wait for completion
        wait (done == 1'b1);
        $display("tb_core: core asserted done, checking %0d outputs...", NEL);

        // compare outputs (o_rdata is combinational)
        errors = 0;
        for (idx = 0; idx < NEL; idx = idx + 1) begin
            o_raddr = idx[ADDR_W-1:0];
            #1;
            if (o_rdata !== o_exp[idx]) begin
                errors = errors + 1;
                if (errors <= 20)
                    $display("  MISMATCH [%0d] got %0d  exp %0d",
                             idx, o_rdata, o_exp[idx]);
            end
        end

        $display("tb_core: %0d elements, %0d errors", NEL, errors);
        if (errors == 0) $display("TB_CORE: PASS");
        else             $display("TB_CORE: FAIL");
        $finish;
    end
endmodule
