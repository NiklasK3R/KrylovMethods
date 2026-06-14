using Test
using LinearAlgebra
using SparseArrays
using KrylovSubspaceMethods

@testset "GMRES Tests" begin

    @testset "Simple nonsysmmetric system" begin
        A = [4.0 1.0 0.0;
             2.0 3.0 1.0;
             0.0 1.0 2.0]
        x_exact = [1.0, 2.0, 3.0]
        b = A * x_exact
        x0 = zeros(3)

        result = gmres(A, b, x0)

        @test result.converged
        @test norm(result.x - x_exact) < 1e-6
    end

    @testset "Sparse nonsymmetric system" begin
        n = 100
        A = spdiagm(-1 => -2*ones(n-1),
                     0  =>  4*ones(n),
                     1  => -1*ones(n-1))
        x_exact = rand(n)
        b = A * x_exact
        x0 = zeros(n)

        result = gmres(A, b, x0)

        @test result.converged
        @test norm(result.x - x_exact) < 1e-6
    end

    @testset "History" begin
        n = 50
        A = spdiagm(-1 => -2*ones(n-1),
                     0  =>  4*ones(n),
                     1  => -1*ones(n-1))
        b = rand(n)
        x0 = zeros(n)

        result = gmres(A, b, x0)

        @test all(diff(result.residuals) .<= 1e-10)
        @test result.residuals[1] ≈ norm(b)
        @test result.residuals[end] < 1e-6
    end

    @testset "Restart" begin
        n = 200
        A = spdiagm(-1 => -2*ones(n-1),
                     0  =>  4*ones(n),
                     1  => -1*ones(n-1))
        x_exact = rand(n)
        b = A * x_exact
        x0 = zeros(n)

        result = gmres(A, b, x0; restart=10, maxiter=500)

        @test result.converged
        @test norm(result.x - x_exact) < 1e-6
    end

    @testset "Max iterations" begin
        n = 1000
        A = spdiagm(-1 => -2*ones(n-1),
                     0  =>  4*ones(n),
                     1  => -1*ones(n-1))
        b = ones(n)
        x0 = zeros(n)

        result = gmres(A, b, x0; maxiter=3, restart=20)

        @test !result.converged
        @test result.iterations == 3
    end

    @testset "Symmetric system (Check against CG)" begin
        n = 50
        A = spdiagm(-1 => -ones(n-1),
                     0 =>  4*ones(n),
                     1 => -ones(n-1))
        x_exact = rand(n)
        b = A * x_exact
        x0 = zeros(n)

        result_gmres = gmres(A, b, x0)
        result_cg    = cg(A, b, x0)

        @test result_gmres.converged
        @test norm(result_gmres.x - x_exact) < 1e-6
        @test norm(result_gmres.x - result_cg.x) < 1e-5
    end

end
