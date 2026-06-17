"""
MINRES: Minimum Residual Method for solving linear systems Ax = b.
#Arguments
#- A::AbstractMatrix: The matrix A in the linear system Ax = b.
#- b::AbstractVector: The right-hand side vector in the linear system Ax = b
#- x0::AbstractVector: The initial guess for the solution.
#- tol::Float64: The tolerance for convergence (default: 1e-6).
#- maxiter::Int: The maximum number of iterations (default: 1000).
#Returns
#- KrylovResult: A struct containing the solution vector, residuals, number of iterations, and convergence status.
"""
function minres(
    A::AbstractMatrix,
    b::AbstractVector,
    x0::AbstractVector;
    tol::Float64 = 1e-6,
    maxiter::Int = 1000,
)
    x = copy(x0)
    r = b - A * x

    p0 = copy(r)
    s0 = A * p0
    p1 = copy(p0)
    s1 = copy(s0)

    residuals = Float64[norm(r)]

    for k = 1:maxiter
        p2 = copy(p1)
        p1 = copy(p0)
        s2 = copy(s1)
        s1 = copy(s0)

        alpha = dot(r, s1) / dot(s1, s1)
        x += alpha * p1
        r -= alpha * s1

        res = norm(r)
        push!(residuals, res)

        if res < tol
            return KrylovResult(x, residuals, k, true)
        end

        p0 = copy(s1)
        s0 = A * s1

        beta = dot(s0, s1) / dot(s1, s1)
        p0 -= beta * p1
        s0 -= beta * s1

        if k > 1
            beta2 = dot(s0, s2) / dot(s2, s2)
            p0 -= beta2 * p2
            s0 -= beta2 * s2
        end
    end

    return KrylovResult(x, residuals, maxiter, false)
end
