using Test
using LinearAlgebra
using SparseArrays
using KrylovSubspaceMethods

@testset "CGW Tests" begin

    @testset "Symmetric case — CGW = CG" begin
        n = 50
        A = spdiagm(-1 => -ones(n-1),
                     0 =>  4*ones(n),
                     1 => -ones(n-1))
        x_exact = rand(n)
        b = A * x_exact
        x0 = zeros(n)

        result_cg  = cg(A, b, x0)
        result_cgw = cgw(A, b, x0)

        @test result_cgw.converged
        @test norm(result_cgw.x - x_exact) < 1e-6
        @test norm(result_cgw.x - result_cg.x) < 1e-6
    end

    @testset "Nonsymmetric positive definite system" begin
        n = 50
        S = spdiagm(-1 => -ones(n-1),
                     0 =>  4*ones(n),
                     1 => -ones(n-1))
        N = 0.1 * sprand(n, n, 0.05)
        A = S + N

        H = 0.5 * (A + A')
        @test minimum(eigvals(Matrix(H))) > 0

        x_exact = rand(n)
        b = A * x_exact
        x0 = zeros(n)

        result = cgw(A, b, x0)

        @test result.converged
        @test norm(result.x - x_exact) < 1e-6
    end

    @testset "History" begin
        n = 50
        S = spdiagm(-1 => -ones(n-1),
                     0 =>  4*ones(n),
                     1 => -ones(n-1))
        N = 0.1 * sprand(n, n, 0.05)
        A = S + N
        H = 0.5 * (A + A')
        if minimum(eigvals(Matrix(H))) > 0
            b = rand(n)
            x0 = zeros(n)
            result = cgw(A, b, x0)

            @test result.residuals[1] ≈ norm(b - A * x0)
            @test result.residuals[end] < 1e-6
        end
    end

    @testset "Maximale Iterationen" begin
        n = 1000
        A = spdiagm(-1 => -ones(n-1),
                     0 =>  2*ones(n),
                     1 => -ones(n-1))
        b = ones(n)
        x0 = zeros(n)

        result = cgw(A, b, x0; maxiter=3)

        @test !result.converged
        @test result.iterations == 3
    end

end
