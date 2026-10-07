# Solvers

Maps neuralls workflow inputs to torchalg solver calls.

## Modules

- `torchalg_runner.py`: `run_traced_pcg` builds an unpreconditioned `torchalg.pcg` call from a dense-or-CSR matrix, right-hand side, initial guess, and `maxiter`/`rtol`/`atol`. It holds no algorithm code; CG, PCG and the preconditioners live in torchalg. Used only by dataset generation (`composition/generation/default_services.py`), not by the comparison workflow — the comparison path's preconditioned solves go through `domain.solver.comparison::run_cg_comparison` instead.
