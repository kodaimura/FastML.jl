using Test

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