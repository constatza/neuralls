# Solvers

Maps neuralls workflow inputs to torchalg solver calls.

## Modules

- `torchalg_runner.py`: builds the `torchalg.pcg` call for a comparison run from the resolved matrix, right-hand side, preconditioner and stopping settings. It holds no algorithm code; CG, PCG and the preconditioners live in torchalg.
