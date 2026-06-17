# Krylov Subspace Methods

A Julia implementation of iterative solvers for linear systems Ax = b using Krylov subspace methods.

## Overview

This package provides implementations of commonly used iterative methods for solving sparse linear systems:
- **CG** - Conjugate Gradient (for symmetric positive-definite systems)
- **MINRES** - Minimum Residual (for symmetric indefinite systems)
- **GMRES** - Generalized Minimum Residual (for non-symmetric systems)
- **CGW** - Conjugate Gradient on the Normal Equations (for non-symmetric systems)

## Usage

### CLI Tool

Solve a linear system using the command-line interface:

```bash
julia bin/solve.jl <matrix> [options]
```

#### Arguments

- `matrix` - Path to the matrix file in Matrix Market format (.mtx)

#### Options

- `-m, --method` - Solver method: `cg`, `cgw`, `minres`, `gmres` (default: `cg`)
- `--tol` - Convergence tolerance (default: 1e-6)
- `--maxiter` - Maximum iterations (default: 1000)
- `--rhs` - Path to right-hand side vector file (one number per line). If not provided, a synthetic system is generated with x_exact = 1

#### Example

```bash
julia bin/solve.jl example/test_matrix.mtx --method cg --tol 1e-6 --maxiter 500
julia bin/solve.jl example/test_matrix.mtx --method gmres --rhs example/rhs.txt
```

## Roadmap

- [ ] Benchmark against standard Julia linear solvers
- [ ] Add BiCG-STAB (BiConjugate Gradient Stabilized)
- [ ] Add preconditioning support
- [ ] Add convergence history plotting
- [ ] Parallel implementation for large-scale systems
