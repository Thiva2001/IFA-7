# IFA-7 — Integer FlashAttention for a Bare-Metal Artix-7 (Nexys A7)

**A fully-integer, retraining-free, online-softmax (FlashAttention-style) attention
microarchitecture for the Xilinx XC7A100T, with a provably non-amplifying
fixed-point softmax and a BRAM-port-aware tile schedule.**

This repository is the implementation of the dissertation proposal *"Resource-Constrained
Attention Acceleration on a Bare-Metal Artix-7 (Nexys A7)"*. It targets the
exact white space identified there: every published hardware FlashAttention is
floating-point and built for large FPGAs/ASICs; every integer-attention work
either targets GPUs or omits FlashAttention tiling. IFA-7 is **PL-only** (no PS,
no DDR/HBM): the whole operator — including softmax — lives in the programmable
logic of a standard student board.

---

## 1. What is in here, and what is verified

| Layer | Artifact | Status in this drop |
|---|---|---|
| Numerical model | `python/` golden model + fixed-point + exp LUT | **Machine-verified here** (`run_tests.py`, all pass) |
| Control logic | `python/ifa7_rtl_emulator.py` (cycle-accurate FSM twin) | **Machine-verified here** (bit-exact vs golden) |
| RTL | `rtl/*.sv` synthesizable SystemVerilog | **Reviewed; elaboration/sim to be run in your Vivado 2024** |
| Testbenches | `tb/*.sv` self-checking vs golden vectors | Provided; run via `vivado/run_sim.tcl` or `sim/run_sim.*` |
| Vivado flow | `vivado/*.tcl`, `constraints/nexys_a7.xdc` | Provided for Vivado 2024 (xc7a100tcsg324-1) |
| Docs/figures | `docs/`, `docs/figures/` | run and try |


---

## 2. The idea in one screen

Standard attention `O = softmax(QKᵀ/√d)·V`. FlashAttention streams key/value
blocks and keeps a running max `m`, running denominator `ℓ`, and output
accumulator `acc`, so the `N×N` score matrix never materialises:

```
m_new  = max(m_old, rowmax(S))           # S = scale(Q·Kᵀ)
alpha  = exp(m_old - m_new)               # ∈ (0,1]  ⇒  NON-AMPLIFYING
p      = exp(S - m_new)                    # clamped tail
ell    = alpha*ell + sum(p)
acc    = alpha*acc + p·V
O      = (acc << OUT_F) / ell              # delayed division (once per row)
```

Everything is **integer / fixed-point** with one global format:
`exp` is `2^(x·log₂e)` realised as **shift + 256-entry LUT** (rounded), the tail
is **clamped**, and the final divide is a **restoring divider**. Because the two
`exp` arguments are always `≤ 0`, every probability is in `(0, 1]` and the
recursion is a contraction — it cannot overflow. This is the methodological core
(see `docs/Architecture.md`).

Verified accuracy vs a double-precision reference (DK=32, N=64, INT8):
**RMS ≈ 0.013, worst element ≈ 0.08** on a ±128 output range — well under the
proposal's ≤1% target.

---

## 3. Quick start

### 3.1 Reproduce the numerical verification (needs only Python + numpy)
```
cd python
py run_tests.py          # full regression: exp, contraction, accuracy, FSM emulator
py gen_vectors.py        # (re)generate rtl/ifa7_pkg.svh, exp LUT, sim vectors
py make_figures.py       # regenerate docs/figures/*.png
```
On Linux/macOS replace `py` with `python3`.

### 3.2 Simulate the RTL (Vivado 2024)
From the Vivado **Tcl console**, with the working directory set to this folder:
```
source vivado/create_project.tcl     ;# build project (xc7a100tcsg324-1)
source vivado/run_sim.tcl            ;# behavioural sim of tb_ifa7_attention_core
```
Look for `TB_CORE: PASS`. To run the other testbenches:
```
set_property top tb_exp_unit       [get_filesets sim_1]; source vivado/run_sim.tcl
set_property top tb_tile_bank_ctrl [get_filesets sim_1]; source vivado/run_sim.tcl
```
Alternatively, from a Vivado-enabled shell: `cd sim && ./run_sim.sh` (or `run_sim.bat`).

### 3.3 Synthesise, implement, bitstream
```
source vivado/create_project.tcl     ;# if not already created
source vivado/build.tcl              ;# synth + impl + bitstream + reports
```
Reports land in `reports/`; the bitstream in
`vivado/ifa7_proj/ifa7_proj.runs/impl_1/nexys_a7_top.bit`.

### 3.4 On-board test (optional)
Program the board, then from the host:
```
cd python
py -c "import numpy as np, ifa7_config as C, host_io; \
       rs=np.random.RandomState(1); \
       Q=rs.randint(-6,7,(C.N,C.DK)); K=rs.randint(-6,7,(C.N,C.DK)); V=rs.randint(-128,128,(C.N,C.DK)); \
       O=host_io.run_on_board(Q,K,V,'COM5'); print(O[:2])"
```
(needs `pip install pyserial`; replace `COM5` with your port).

---

## 4. Repository layout
See `docs/ProjectStructure.md`. Short version:
```
python/        golden model, fixed-point, exp LUT, FSM emulator, vector + figure gen
rtl/           synthesizable SystemVerilog (the design)
tb/            self-checking SystemVerilog testbenches
sim/           command-line xsim scripts + generated vectors/
vivado/        Vivado 2024 Tcl: create_project / run_sim / build
constraints/   nexys_a7.xdc
docs/          Architecture, VerificationReport, BuildGuide, ProjectStructure, figures
reports/       (populated by the Vivado build)
```

## 5. Documentation
- `docs/Architecture.md` — microarchitecture, fixed-point numerics, error/overflow analysis, scheduling.
- `docs/Architecture.pdf` — publication-style summary + figures (regenerate: `py python/make_report_pdf.py`).
- `docs/VerificationReport.md` — exactly what was tested, results, and what remains for hardware.
- `docs/BuildGuide.md` — detailed build/run/serial instructions.
- `docs/ProjectStructure.md` — every file explained.

## 6. License / citation
Open research artifact accompanying the IFA-7 dissertation work. If you use it,
please cite the dissertation proposal and this repository.
