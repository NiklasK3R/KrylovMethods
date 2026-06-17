"""
CGW Method to solve Ax = b, where A is a non-symmetric positive definite matrix.
This method utilizes CG to solve the inner system involving the symmetrized matrix H = (A + A') / 2.
# Arguments
# - `A`: Coefficient matrix (non-symmetric positive definite)
# - `b`: Right-hand side vector
# - `x0`: Initial guess for the solution
# - `tol`: Tolerance for convergence (default: 1e-6)
# - `maxiter`: Maximum number of iterations (default: 1000)
# Returns
# - `KrylovResult`: A struct containing the solution, residuals, number of
#    iterations, and convergence status.
"""

function cgw(
    A::AbstractMatrix,
    b::AbstractVector,
    x0::AbstractVector;
    tol::Float64 = 1e-6,
    maxiter::Int = 1000,
)
    x = copy(x0)
    r = b - A * x
    H = symmetrize(A)
    H_inv_r = cg(H, r, zeros(size(r)); tol = tol, maxiter = maxiter).x
    p = copy(H_inv_r)

    residuals = Float64[]
    push!(residuals, norm(r))

    for k = 1:maxiter
        rHr = dot(r, H_inv_r)
        alpha = rHr / dot(p, A * p)

        x += alpha * p
        r -= alpha * A * p
        res = norm(r)
        push!(residuals, res)

        if res < tol
            return KrylovResult(x, residuals, k, true)
        end

        H_inv_r = cg(H, r, zeros(length(r)); tol = tol, maxiter = maxiter).x
        beta = -dot(r, H_inv_r) / rHr
        p = H_inv_r + beta * p
    end

    return KrylovResult(x, residuals, maxiter, false)
end
