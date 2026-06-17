using Test
using LinearAlgebra
using SparseArrays
using KrylovSubspaceMethods

@testset "MINRES Tests" begin

    @testset "Simple SPD System (Sanity-Check gegen CG)" begin
        n = 50
        A = spdiagm(-1 => -ones(n-1),
                     0 =>  4*ones(n),
                     1 => -ones(n-1))
        x_exact = rand(n)
        b = A * x_exact
        x0 = zeros(n)

        result_minres = minres(A, b, x0)
        result_cg     = cg(A, b, x0)

        @test result_minres.converged
        @test norm(result_minres.x - x_exact) < 1e-6
        @test norm(result_minres.x - result_cg.x) < 1e-5
    end

    @testset "Symmetric indefinite System" begin
        n = 50
        eigvals = vcat(-rand(25) .- 1.0, rand(25) .+ 1.0)
        Q, _ = qr(randn(n, n))
        Q = Matrix(Q)
        A_sym = Q * Diagonal(eigvals) * Q'
        A_sym = Symmetric(A_sym) |> Matrix

        x_exact = rand(n)
        b = A_sym * x_exact
        x0 = zeros(n)

        result = minres(A_sym, b, x0; maxiter=200)

        @test result.converged
        @test norm(result.x - x_exact) < 1e-5
    end

    @testset "Sparse symmetric System" begin
        n = 100
        A = spdiagm(-1 => -ones(n-1),
                     0 =>  4*ones(n),
                     1 => -ones(n-1))
        x_exact = rand(n)
        b = A * x_exact
        x0 = zeros(n)

        result = minres(A, b, x0)

        @test result.converged
        @test norm(result.x - x_exact) < 1e-6
    end

    @testset "History" begin
        n = 50
        A = spdiagm(-1 => -ones(n-1),
                     0 =>  4*ones(n),
                     1 => -ones(n-1))
        b = rand(n)
        x0 = zeros(n)

        result = minres(A, b, x0)

        @test isapprox(result.residuals[1], norm(b))
        @test result.residuals[end] < 1e-6
    end

    @testset "Max Iterations" begin
        n = 1000
        A = spdiagm(-1 => -ones(n-1),
                     0 =>  2*ones(n),
                     1 => -ones(n-1))
        b = ones(n)
        x0 = zeros(n)

        result = minres(A, b, x0; maxiter=3)

        @test !result.converged
        @test result.iterations == 3
    end

end
