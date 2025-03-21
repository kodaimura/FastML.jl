using FastML
using Test
using Plots

@testset "FastML.jl" begin
    f(x) = 3x + 5
    data = [([x + rand(Float32)], f(x) + rand(Float32)) for x in -3:0.1f0:3]
    x = hcat([d[1] for d in data]...)
    y = hcat([d[2] for d in data]...)

    model = LinearRegression(1;learning_rate=0.01,max_epochs=10000,tolerance=1e-6)
    train!(model, data)
    predicted_values = predict(model, x)

    m = get_model(model)
    plot(vec(x), vec(y), seriestype = :scatter, label="True values", title="Model Training")
    plot!((x) -> m.bias[1] + m.weight[1] * x, label="After Training", lw=2)

    savefig("linear_regression.png")
end
