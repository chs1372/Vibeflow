# Zero-pressure-gradient flat plate (ADR-042, gate 4)

NASA Turbulence Modeling Resource's 2DZP verification case, run with the
k-ω SST model.

- `tmr/*.p2dfmt.gz` — TMR's five nested grids, 35×25 to 545×385
  (https://tmbwg.github.io/turbmodels/flatplate_grids.html).
- `tmr/*_sstv.dat` — TMR's SST-V results from CFL3D and FUN3D: Cf at
  x = 0.97 against grid size, the drag, Cf along the plate, u⁺ profiles at
  x = 0.97008 and 1.90334, ν_t at x = 0.97
  (https://tmbwg.github.io/turbmodels/flatplate_sst.html).
- `make_mesh.py` writes each grid as a one-cell-thick `.hex` slab beside
  this file (ignored by git).

Run the gate with `python3 tests/benchmark/flat_plate_gate.py`; a single
grid with `build/tests/flat_plate cases/flatplate/<level>.hex <out prefix>
[1994|2003] [dt] [max steps]`.
