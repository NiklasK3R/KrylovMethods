"""
Performs one step of the Arnoldi process to expand the Krylov subspace.
#Arguments
#- `A`: The matrix of the linear system.
#- `V`: The matrix whose columns are the orthonormal basis vectors of the Krylov subspace.
#- `H`: The upper Hessenberg matrix that will be updated.
#- `k`: The current step index.
#Returns
#- Nothing. The function modifies `V` and `H` in place.
"""
function _arnoldi_step!(A::AbstractMatrix, V::AbstractMatrix, H::AbstractMatrix, k::Int)
    w = A * V[:, k]

    for j = 1:k
        H[j, k] = dot(V[:, j], w)
        w -= H[j, k] * V[:, j]
    end

    H[k+1, k] = norm(w)

    if H[k+1, k] > 1e-12
        V[:, k+1] = w / H[k+1, k]
    end
    return nothing
end

"""
GMRES (Generalized Minimal Residual) method for solving linear systems Ax = b.
# Arguments
#- `A`: The matrix of the linear system.
#- `b`: The right-hand side vector.
#- `x0`: The initial guess for the solution.
#- `tol`: The tolerance for convergence (default: 1e-6).
#- `maxiter`: The maximum number of iterations (default: 1000).
#- `restart`: The number of iterations after which the method restarts (default: 20).
# Returns
#- A `KrylovResult` object containing the solution, residuals, total iterations, and convergence status.
"""
function gmres(
    A::AbstractMatrix,
    b::AbstractVector,
    x0::AbstractVector;
    tol::Float64 = 1e-6,
    maxiter::Int = 1000,
    restart::Int = 20,
)
    n = length(b)
    x = copy(x0)
    residuals = Float64[]
    total_iters = 0

    while total_iters < maxiter
        r = b - A * x
        beta = norm(r)
        push!(residuals, beta)

        if beta < tol
            return KrylovResult(x, residuals, total_iters, true)
        end

        m = min(restart, maxiter - total_iters)

        V = zeros(n, m + 1)
        H = zeros(m + 1, m)
        V[:, 1] = r / beta

        y = Float64[]
        k_old = 0

        for k = 1:m
            _arnoldi_step!(A, V, H, k)
            total_iters += 1
            k_old = k

            e1 = zeros(k + 1)
            e1[1] = beta
            y = H[1:(k+1), 1:k] \ e1

            res = norm(H[1:(k+1), 1:k] * y - e1)
            push!(residuals, res)

            if res < tol
                x += V[:, 1:k] * y
                return KrylovResult(x, residuals, total_iters, true)
            end

            if H[k+1, k] <= 1e-12
                break
            end
        end
        x += V[:, 1:k_old] * y
    end
    return KrylovResult(x, residuals, total_iters, false)
end
