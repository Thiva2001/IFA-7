
`timescale 1ns/1ps
`ifndef IFA7_VEC_DIR
  `define IFA7_VEC_DIR "vectors/"
`endif

module tb_exp_unit;
    logic signed [31:0] x;
    logic        [16:0] p;

    exp_unit dut (.x(x), .p(p));

    integer fd, r, errors, total;
    integer xi, pe;

    initial begin
        errors = 0; total = 0;
        fd = $fopen({`IFA7_VEC_DIR, "exp_test.mem"}, "r");
        if (fd == 0) begin
            $display("FATAL: cannot open %sexp_test.mem", `IFA7_VEC_DIR);
            $fatal;
        end
        while (!$feof(fd)) begin
            r = $fscanf(fd, "%h %h", xi, pe);
            if (r == 2) begin
                x = xi;
                #1;                                  // settle combinational path
                total = total + 1;
                if (p !== pe[16:0]) begin
                    errors = errors + 1;
                    if (errors <= 20)
                        $display("  MISMATCH x=%0d  got p=%0d  exp=%0d",
                                 $signed(x), p, pe[16:0]);
                end
            end
        end
        $fclose(fd);
        $display("tb_exp_unit: %0d vectors checked, %0d errors", total, errors);
        if (errors == 0) $display("TB_EXP_UNIT: PASS");
        else             $display("TB_EXP_UNIT: FAIL");
        $finish;
    end
endmodule
