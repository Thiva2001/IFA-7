// ============================================================================
// uart_tx.sv  --  8N1 UART transmitter (PL-only stream front-end)
// ----------------------------------------------------------------------------
// Assert `start` for one clock with `data` valid while `busy` is low.  `busy`
// stays high until the stop bit completes.  DIV = system_clock / baud.
// ============================================================================
`timescale 1ns/1ps
`default_nettype none

module uart_tx #(
    parameter int DIV = 868
) (
    input  wire       clk,
    input  wire       rstn,
    input  wire       start,
    input  wire [7:0] data,
    output reg        tx,
    output reg        busy
);
    localparam int CW = $clog2(DIV);

    typedef enum logic [1:0] {IDLE, START, DATA, STOP} st_t;
    st_t st;
    reg [CW-1:0] cnt;
    reg [2:0]    bit_i;
    reg [7:0]    sh;

    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            st <= IDLE; tx <= 1'b1; busy <= 1'b0; cnt <= '0; bit_i <= '0; sh <= '0;
        end else begin
            case (st)
                IDLE: begin
                    tx <= 1'b1; busy <= 1'b0;
                    if (start) begin sh <= data; busy <= 1'b1; st <= START; cnt <= '0; end
                end
                START: begin
                    tx <= 1'b0; busy <= 1'b1;
                    if (cnt == CW'(DIV-1)) begin cnt <= '0; st <= DATA; bit_i <= '0; end
                    else cnt <= cnt + 1'b1;
                end
                DATA: begin
                    tx <= sh[0]; busy <= 1'b1;
                    if (cnt == CW'(DIV-1)) begin
                        cnt <= '0; sh <= {1'b0, sh[7:1]};
                        if (bit_i == 3'd7) st <= STOP;
                        else bit_i <= bit_i + 1'b1;
                    end else cnt <= cnt + 1'b1;
                end
                STOP: begin
                    tx <= 1'b1; busy <= 1'b1;
                    if (cnt == CW'(DIV-1)) begin cnt <= '0; st <= IDLE; busy <= 1'b0; end
                    else cnt <= cnt + 1'b1;
                end
                default: st <= IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
