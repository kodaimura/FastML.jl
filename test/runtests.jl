using Test
using FastML

@testset "FastML.jl" begin
    @testset "linear_regression.jl" begin
        include("linear_regression.jl")
    end
end
