"""
KrylovResult
-x: The solution vector obtained from the Krylov subspace method.
-residuals: A vector containing the residual norms at each iteration of the Krylov method.
-iterations: The number of iterations taken to reach convergence or the maximum number of iterations.
-converged: A boolean indicating whether the method converged within the specified tolerance.
"""

struct KrylovResult
  x::Vector{Float64}
  residuals::Vector{Float64}
  iterations::Int
  converged::Bool
end

"""
Returns the symmetric part of A: H:= 0.5*(A + A'), used by CGW.
"""
function symmetrize(A::AbstractMatrix)
  return 0.5 * (A + A')
end

function converged(r::Vector{Float64}, tol::Float64)
  return norm(r) < tol
end

