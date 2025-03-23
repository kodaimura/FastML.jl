using FastML
using Test
using Plots

@testset "FastML.jl" begin
    f(x) = 3x + 5
    data = [([x + rand(Float32)], f(x) + rand(Float32)) for x in -3:0.1f0:3]
    x = hcat([d[1] for d in data]...)
    y = hcat([d[2] for d in data]...)

    x_train, y_train, x_test, y_test = split_data(x, y, 0.2)

    model = LinearRegression(1;learning_rate=0.01,max_epochs=10000,tolerance=1e-6)
    train!(model, x_train, y_train)

    println(r2(model, x_train, y_train))
    println(r2(model, x_test, y_test))

    w = weight(model)
    b = bias(model)

    plot(vec(x), vec(y), seriestype = :scatter, label="True values", title="Model Training")
    plot!((x) -> b[1] + w[1] * x, label="After Training", lw=2)

    savefig("linear_regression.png")
end
