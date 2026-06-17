using Test
using KrylovSubspaceMethods

@testset "KrylovSubspaceMethods.jl" begin
    include("test_cg.jl")
    include("test_cgw.jl")
    include("test_minres.jl")
    include("test_gmres.jl")
end
