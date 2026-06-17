#!/usr/bin/env julia

using ArgParse
using MatrixMarket
using LinearAlgebra
using KrylovSubspaceMethods

function parse_commandline()
    s = ArgParseSettings(
        description = "Solve a linear system Ax = b using iterative methods.",
    )

    @add_arg_table! s begin
        "matrix"
        help = "Path to .mtx matrix file"
        required = true
        "--method", "-m"
        help = "cg, cgw, minres, gmres"
        default = "cg"
        "--tol"
        help = "Convergence tolerance"
        arg_type = Float64
        default = 1e-6
        "--maxiter"
        help = "Maximum iterations"
        arg_type = Int
        default = 1000
        "--rhs"
        help = "Path to b (one number per line). If empty: synthetic b with x_exact=1"
        default = ""
    end

    return parse_args(s)
end

function load_rhs(path::String, n::Int)
    if isempty(path)
        x_exact = ones(n)
        return x_exact, nothing
    else
        b = parse.(Float64, readlines(path))
        return nothing, b
    end
end

function main()
    args = parse_commandline()

    println("Reading matrix: $(args["matrix"])")
    A = MatrixMarket.mmread(args["matrix"])
    n = size(A, 1)

    x_exact, b = load_rhs(args["rhs"], n)
    if b === nothing
        b = A * x_exact
        println("No b specified — synthetic system with x_exact = 1 generated.")
    end

    x0 = zeros(n)

    method = args["method"]
    tol = args["tol"]
    maxiter = args["maxiter"]

    println("Solving with method: $method (tol=$tol, maxiter=$maxiter)")

    result = if method == "cg"
        cg(A, b, x0; tol = tol, maxiter = maxiter)
    elseif method == "cgw"
        cgw(A, b, x0; tol = tol, maxiter = maxiter)
    elseif method == "minres"
        minres(A, b, x0; tol = tol, maxiter = maxiter)
    elseif method == "gmres"
        gmres(A, b, x0; tol = tol, maxiter = maxiter)
    else
        error("Unknown method: $method")
    end

    println()
    println("Converged: $(result.converged)")
    println("Iterations: $(result.iterations)")
    println("Final residual: $(result.residuals[end])")
    println("Final solution vector x: $(result.x)")

    if x_exact !== nothing
        err = norm(result.x - x_exact)
        println("Error ‖x - x_exact‖: $err")
    end
end

main()
