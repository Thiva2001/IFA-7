@echo off
REM ===========================================================================
REM run_sim.bat  --  Command-line xsim driver for all IFA-7 testbenches (Windows)
REM ---------------------------------------------------------------------------
REM Run from the sim\ directory in a Vivado-enabled command prompt
REM (xvlog/xelab/xsim on PATH; e.g. after running Vivado's settings64.bat).
REM
REM     cd sim
REM     run_sim.bat
REM ===========================================================================
setlocal
cd /d "%~dp0"
set RTL=..\rtl
set TB=..\tb

echo === compiling RTL + testbenches ===
call xvlog -sv -i "%RTL%" ^
  "%RTL%\exp_lut.sv" "%RTL%\exp_unit.sv" "%RTL%\divider.sv" ^
  "%RTL%\ifa7_attention_core.sv" "%RTL%\tile_bank_ctrl.sv" ^
  "%RTL%\uart_rx.sv" "%RTL%\uart_tx.sv" "%RTL%\frame_ctrl_fsm.sv" "%RTL%\nexys_a7_top.sv" ^
  "%TB%\tb_exp_unit.sv" "%TB%\tb_tile_bank_ctrl.sv" "%TB%\tb_ifa7_attention_core.sv" "%TB%\tb_frame_core.sv"
if errorlevel 1 goto :err

for %%T in (tb_exp_unit tb_tile_bank_ctrl tb_ifa7_attention_core tb_frame_core) do (
  echo === elaborate + simulate: %%T ===
  call xelab -debug off %%T -s sim_%%T
  if errorlevel 1 goto :err
  call xsim sim_%%T -runall
  if errorlevel 1 goto :err
)
echo === done. Search the output above for PASS / FAIL. ===
goto :eof

:err
echo *** simulation flow failed ***
exit /b 1
