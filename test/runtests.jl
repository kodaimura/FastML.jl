using Test

include("test_common.jl")

@testset "FastML.jl" begin
    @testset "utils" begin
        include("utils.jl")
    end

    @testset "models" begin
        include("models.jl")
    end

    @testset "train!" begin
        include("train.jl")
    end
end