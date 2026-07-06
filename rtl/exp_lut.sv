
`timescale 1ns/1ps
`default_nettype none

module exp_lut #(
    parameter int LUT_BITS = 8,
    parameter int LUT_W    = 18
) (
    input  wire [LUT_BITS-1:0] addr,
    output wire [LUT_W-1:0]    data
);
    localparam int DEPTH = (1 << LUT_BITS);

    // ROM storage. Initialised once at elaboration from the generated include.
    // Asynchronous (combinational) read -> distributed (LUT) ROM.
    (* rom_style = "distributed" *)
    logic [LUT_W-1:0] rom [0:DEPTH-1];

    initial begin
        `include "exp_lut_init.svh"
    end

    assign data = rom[addr];
endmodule

`default_nettype wire
