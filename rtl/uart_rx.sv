// ============================================================================
// uart_rx.sv  --  8N1 UART receiver (PL-only stream front-end)
// ----------------------------------------------------------------------------
// Standard mid-bit sampling receiver.  DIV = system_clock / baud (default 868
// for 100 MHz / 115200).  `valid` pulses for one clock when `data` is updated.
// ============================================================================
`timescale 1ns/1ps
`default_nettype none

module uart_rx #(
    parameter int DIV = 868
) (
    input  wire       clk,
    input  wire       rstn,
    input  wire       rx,        // asynchronous serial input
    output reg  [7:0] data,
    output reg        valid      // 1-cycle pulse
);
    localparam int CW   = $clog2(DIV);
    localparam int HALF = DIV / 2;

    // two-FF synchroniser (idle line is high)
    reg [1:0] sync;
    wire      rxs = sync[1];
    always_ff @(posedge clk or negedge rstn)
        if (!rstn) sync <= 2'b11;
        else       sync <= {sync[0], rx};

    typedef enum logic [1:0] {IDLE, START, DATA, STOP} st_t;
    st_t st;
    reg [CW-1:0] cnt;
    reg [2:0]    bit_i;
    reg [7:0]    sh;

    always_ff @(posedge clk or negedge rstn) begin
        if (!rstn) begin
            st <= IDLE; valid <= 1'b0; cnt <= '0; bit_i <= '0; sh <= '0; data <= '0;
        end else begin
            valid <= 1'b0;
            case (st)
                IDLE:  if (!rxs) begin st <= START; cnt <= '0; end
                START: if (cnt == CW'(HALF-1)) begin
                           if (!rxs) begin st <= DATA; cnt <= '0; bit_i <= '0; end
                           else        st <= IDLE;          // false start, reject
                       end else cnt <= cnt + 1'b1;
                DATA:  if (cnt == CW'(DIV-1)) begin
                           cnt <= '0; sh <= {rxs, sh[7:1]}; // LSB first
                           if (bit_i == 3'd7) st <= STOP;
                           else bit_i <= bit_i + 1'b1;
                       end else cnt <= cnt + 1'b1;
                STOP:  if (cnt == CW'(DIV-1)) begin
                           st <= IDLE; data <= sh; valid <= 1'b1;
                       end else cnt <= cnt + 1'b1;
                default: st <= IDLE;
            endcase
        end
    end
endmodule

`default_nettype wire
