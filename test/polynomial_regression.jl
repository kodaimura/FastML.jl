using Test
using FastML
using Plots

gr()

@testset "polynomial_regression" begin
    f(x) = 2 + 3x + 5x^2 - 3x^3
    data = [([x + rand(Float32)], f(x) + rand(Float32)) for x in -3:0.1f0:3]
    x = reduce(hcat, first.(data))
    y = reduce(hcat, last.(data))

    x_train, y_train, x_test, y_test = split_data(x, y, 0.2)
    model = FastML.PolynomialRegression.Model(3;learning_rate=0.0003,max_epochs=10000,tolerance=1e-6)
    FastML.PolynomialRegression.train!(model, x_train, y_train)

    @test 1 > FastML.PolynomialRegression.r2(model, x_train, y_train) > 0.6
    @test 1 > FastML.PolynomialRegression.r2(model, x_test, y_test) > 0.6

    w = FastML.PolynomialRegression.weight(model)
    b = FastML.PolynomialRegression.bias(model)
    Plots.plot(vec(x), vec(y), seriestype=:scatter, label="True values", title="Polynomial Regression")
    Plots.plot!((x) -> b[1] + w[3] * x^3 + w[2] * x^2 + w[1] * x, label="Predicted values", lw=2)
    Plots.savefig("polynomial_regression.png")
end