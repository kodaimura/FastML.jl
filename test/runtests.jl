using Test

@testset "FastML.jl" begin
    @testset "utils" begin
        include("utils.jl")
    end

    @testset "models" begin
        include("models.jl")
    end
end