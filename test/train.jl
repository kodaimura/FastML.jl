using Test
using Flux
using FastML
using Plots
using Random

@testset "train!" begin
    f(x) = 3x + 5
    X = collect(-3:0.1f0:3) .+ rand(Float32, length(-3:0.1f0:3))
    y = f.(X) .+ rand(Float32, length(X))

    X = reshape(X, 61, 1)'
    y = reshape(y, 61, 1)'

    X_train, y_train, X_test, y_test = split_train_test(X, y; test_size=0.2, shuffle=false)

    model = Flux.Dense(1 => 1)
    trainer = Trainer(:linear_regression)
    done, epoch, loss = train!(model, X_train, y_train, trainer)

    #Plots.plot(vec(X), vec(y), seriestype = :scatter, label="True values", title="Model Training")
    #Plots.plot!((x) -> model.bias[1] + model.weight[1] * x, label="After Training", lw=2)
    #Plots.savefig("linear_regression.png")
end