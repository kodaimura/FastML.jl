using Test
using FastML
using Plots

gr()

@testset "linear_regression" begin
    f(x) = 3x + 5
    data = [([x + rand(Float32)], f(x) + rand(Float32)) for x in -3:0.1f0:3]
    x = reduce(hcat, first.(data))
    y = reduce(hcat, last.(data))

    x_train, y_train, x_test, y_test = split_data(x, y, 0.2)
    model = FastML.LinearRegression.Model(1;learning_rate=0.01,max_epochs=10000,tolerance=1e-6)
    FastML.LinearRegression.train!(model, x_train, y_train)

    @test 1 > FastML.LinearRegression.r2(model, x_train, y_train) > 0.9
    @test 1 > FastML.LinearRegression.r2(model, x_test, y_test) > 0.9

    w = FastML.LinearRegression.weight(model)
    b = FastML.LinearRegression.bias(model)
    Plots.plot(vec(x), vec(y), seriestype = :scatter, label="True values", title="Model Training")
    Plots.plot!((x) -> b[1] + w[1] * x, label="After Training", lw=2)
    Plots.savefig("linear_regression.png")
end

@testset "linear_regression2" begin
    f(x) = 3x[1] + 2x[2] - x[3] + 4x[4] - 2x[5] + 1
    data = []
    for i in 1:100
        xi = [rand(Float32), rand(Float32), rand(Float32), rand(Float32), rand(Float32)]
        yi = f(xi) + rand(Float32)
        push!(data, (xi, yi))
    end
    x = reduce(hcat, first.(data))
    y = reduce(hcat, last.(data))

    x_train, y_train, x_test, y_test = split_data(x, y, 0.2)
    model = FastML.LinearRegression.Model(5;learning_rate=0.01,max_epochs=10000,tolerance=1e-6)
    FastML.LinearRegression.train!(model, x_train, y_train)

    @test 1 > FastML.LinearRegression.r2(model, x_train, y_train) > 0.9
    @test 1 > FastML.LinearRegression.r2(model, x_test, y_test) > 0.9
end