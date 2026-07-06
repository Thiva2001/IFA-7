
`timescale 1ns/1ps
`default_nettype none

module divider #(
    parameter int DW = 56,   // dividend / quotient width
    parameter int VW = 40    // divisor width
) (
    input  wire           clk,
    input  wire           rstn,
    input  wire           start,    // 1-cycle pulse
    input  wire [DW-1:0]  dividend,
    input  wire [VW-1:0]  divisor,
    output reg            busy,
    output reg            done,     // 1-cycle pulse when result valid
    output reg  [DW-1:0]  quotient,
    output reg  [VW-1:0]  remainder
);
    localparam int CW = $clog2(DW + 1);

    reg [DW-1:0]  d_reg;      // dividend being shifted out (MSB first)
    reg [DW-1:0]  q_reg;      // quotient being shifted in (LSB)
    reg [VW:0]    r_reg;      // running remainder (one guard bit)
    reg [VW-1:0]  v_reg;      // latched divisor
    reg [CW-1:0]  cnt;

    // Next remainder after bringing down one dividend bit.
    wire [VW:0] r_shift = {r_reg[VW-1:0], d_reg[DW-1]};
    wire        ge      = (r_shift >= {1'b0, v_reg});
    wire [VW:0] r_next  = ge ? (r_shift - {1'b0, v_reg}) : r_shift;

    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            busy <= 1'b0; done <= 1'b0;
            quotient <= '0; remainder <= '0;
            d_reg <= '0; q_reg <= '0; r_reg <= '0; v_reg <= '0; cnt <= '0;
        end else begin
            done <= 1'b0;
            if (start && !busy) begin
                if (divisor == '0) begin
                    // divide-by-zero guard: define result 0 (cannot occur in
                    // IFA-7 because the row max always contributes p = 1.0).
                    quotient <= '0; remainder <= '0; done <= 1'b1; busy <= 1'b0;
                end else begin
                    d_reg <= dividend; q_reg <= '0; r_reg <= '0;
                    v_reg <= divisor;  cnt <= CW'(DW); busy <= 1'b1;
                end
            end else if (busy) begin
                r_reg <= r_next;
                q_reg <= {q_reg[DW-2:0], ge};
                d_reg <= {d_reg[DW-2:0], 1'b0};
                cnt   <= cnt - 1'b1;
                if (cnt == CW'(1)) begin
                    busy      <= 1'b0;
                    done      <= 1'b1;
                    quotient  <= {q_reg[DW-2:0], ge};
                    remainder <= r_next[VW-1:0];
                end
            end
        end
    end
endmodule

`default_nettype wire
