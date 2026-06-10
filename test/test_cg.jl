using Test
using LinearAlgebra
using SparseArrays
using KrylovSubspaceMethods

@testset "CG Tests" begin

    @testset "Simple SPD System" begin
        A = [4.0 1.0 0.0;
             1.0 3.0 1.0;
             0.0 1.0 2.0]
        x_exact = [1.0, 2.0, 3.0]
        b = A * x_exact
        x0 = zeros(3)

        result = cg(A, b, x0)

        @test result.converged
        @test norm(result.x - x_exact) < 1e-6
    end

    @testset "Sparse SPD System" begin
        n = 100
        A = spdiagm(-1 => -ones(n-1),
                     0 =>  4*ones(n),
                     1 => -ones(n-1))
        x_exact = rand(n)
        b = A * x_exact
        x0 = zeros(n)

        result = cg(A, b, x0)

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

        result = cg(A, b, x0)

        @test all(diff(result.residuals) .<= 1e-14)
        @test result.residuals[1] ≈ norm(b)
    end

    @testset "Maximale Iterationen" begin
        n = 1000
        A = spdiagm(-1 => -ones(n-1),
                     0 =>  2*ones(n),
                     1 => -ones(n-1))
        b = ones(n)
        x0 = zeros(n)

        result = cg(A, b, x0; maxiter=5)

        @test !result.converged
        @test result.iterations == 5
    end

end
