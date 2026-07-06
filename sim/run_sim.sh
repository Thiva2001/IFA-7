#!/usr/bin/env bash
# ============================================================================
# run_sim.sh  --  Command-line xsim driver for all IFA-7 testbenches
# ----------------------------------------------------------------------------
# Run from the sim/ directory in a Vivado-enabled shell (xvlog/xelab/xsim on
# PATH; e.g. after sourcing Vivado's settings64.sh). The testbenches read the
# generated vectors from ./vectors (default IFA7_VEC_DIR).
#
#     cd sim && ./run_sim.sh
# ============================================================================
set -e
cd "$(dirname "$0")"
RTL=../rtl
TB=../tb

echo "=== regenerating vectors (optional; needs python) ==="
( cd ../python && py gen_vectors.py ) || echo "  (skipped vector regen)"

echo "=== compiling RTL + testbenches ==="
xvlog -sv -i "$RTL" \
  "$RTL/exp_lut.sv" "$RTL/exp_unit.sv" "$RTL/divider.sv" \
  "$RTL/ifa7_attention_core.sv" "$RTL/tile_bank_ctrl.sv" \
  "$RTL/uart_rx.sv" "$RTL/uart_tx.sv" "$RTL/frame_ctrl_fsm.sv" "$RTL/nexys_a7_top.sv" \
  "$TB/tb_exp_unit.sv" "$TB/tb_tile_bank_ctrl.sv" \
  "$TB/tb_ifa7_attention_core.sv" "$TB/tb_frame_core.sv"

run_one () {
  echo "=== elaborate + simulate: $1 ==="
  xelab -debug off "$1" -s "sim_$1"
  xsim "sim_$1" -runall
}

run_one tb_exp_unit
run_one tb_tile_bank_ctrl
run_one tb_ifa7_attention_core
run_one tb_frame_core

echo "=== done. Search the output above for 'PASS' / 'FAIL'. ==="
