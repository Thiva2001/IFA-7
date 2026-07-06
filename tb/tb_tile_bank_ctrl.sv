// ============================================================================
// tb_tile_bank_ctrl.sv  --  functional check for the port-aware ping-pong bank
// ----------------------------------------------------------------------------
// Demonstrates that the producer (write, port A, fill bank) and the consumer
// (read, port B, drain bank) operate on different banks with no contention, and
// that data written before a swap is readable after it.
// ============================================================================
`timescale 1ns/1ps

module tb_tile_bank_ctrl;
    localparam int DW = 32, DEPTH = 16, AW = $clog2(16);

    logic           clk = 0, swap, wr_en, phase_o;
    logic [AW-1:0]  wr_addr, rd_addr;
    logic [DW-1:0]  wr_data, rd_data;

    tile_bank_ctrl #(.DW(DW), .DEPTH(DEPTH)) dut (
        .clk(clk), .swap(swap), .wr_en(wr_en), .wr_addr(wr_addr),
        .wr_data(wr_data), .rd_addr(rd_addr), .rd_data(rd_data),
        .phase_o(phase_o) );

    always #5 clk = ~clk;

    integer i, errors;
    initial begin
        errors = 0; swap = 0; wr_en = 0; wr_addr = 0; wr_data = 0; rd_addr = 0;
        @(negedge clk);

        // Phase 0: fill bank0 with addr+100
        for (i = 0; i < 8; i = i + 1) begin
            @(negedge clk); wr_en = 1; wr_addr = i[AW-1:0]; wr_data = i + 100;
        end
        @(negedge clk); wr_en = 0;

        // swap -> drain = bank0
        @(negedge clk); swap = 1; @(negedge clk); swap = 0;

        // read back bank0 (1-cycle latency) AND fill bank1 with addr+200
        for (i = 0; i < 8; i = i + 1) begin
            @(negedge clk);
            rd_addr = i[AW-1:0];
            wr_en = 1; wr_addr = i[AW-1:0]; wr_data = i + 200;
            @(negedge clk);                       // read latency
            #1;
            if (rd_data !== (i + 100)) begin
                errors = errors + 1;
                $display("  MISMATCH drain bank0 [%0d] got %0d exp %0d",
                         i, rd_data, i + 100);
            end
        end
        wr_en = 0;

        // swap -> drain = bank1
        @(negedge clk); swap = 1; @(negedge clk); swap = 0;
        for (i = 0; i < 8; i = i + 1) begin
            @(negedge clk); rd_addr = i[AW-1:0];
            @(negedge clk); #1;
            if (rd_data !== (i + 200)) begin
                errors = errors + 1;
                $display("  MISMATCH drain bank1 [%0d] got %0d exp %0d",
                         i, rd_data, i + 200);
            end
        end

        $display("tb_tile_bank_ctrl: %0d errors", errors);
        if (errors == 0) $display("TB_TILE_BANK: PASS");
        else             $display("TB_TILE_BANK: FAIL");
        $finish;
    end
endmodule
