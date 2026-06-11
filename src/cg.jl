"""
Classic Conjugate Gradient (CG) method for solving linear systems of the form Ax = b, where A is a symmetric positive-definite matrix.
# Arguments
# - `A`: Coefficient matrix (symmetric positive-definite)
# - `b`: Right-hand side vector
# - `x0`: Initial guess for the solution
# - `tol`: Tolerance for convergence (default: 1e-6)
# - `maxiter`: Maximum number of iterations (default: 1000)
# Returns
# - `KrylovResult`: A struct containing the solution, residuals, number of iterations, and convergence status.
"""

function cg(A::AbstractMatrix, b::AbstractVector, x0::AbstractVector; tol::Float64=1e-6, maxiter::Int=1000)
  x = copy(x0)
  r = b - A * x
  p = copy(r)

  residuals = Float64[]
  push!(residuals, norm(r))

  for k in 1:maxiter
    Ap = A * p
    rHr = dot(r, r)
    alpha = rHr / dot(p, Ap)

    x += alpha * p
    r -= alpha * Ap
    res = norm(r)

    push!(residuals, res)


    if res < tol
      return KrylovResult(x, residuals, k, true)
    end

    omega = dot(r, r) / rHr
    p = r + omega * p
  end

  return KrylovResult(x, residuals, maxiter, false)
end

