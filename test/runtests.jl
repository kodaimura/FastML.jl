using Test
using FastML
using Flux
using Random
using Statistics
using Plots

@testset "FastML.jl" begin
    @testset "utils" begin
        include("utils.jl")
    end

    @testset "models" begin
        include("models.jl")
    end

    @testset "train!" begin
        include("training.jl")
    end
end
