using Test
using Flux
using FastML
using Plots
using Random

@testset "train!" begin
    @testset "linear_regression" begin
        X, y = sample_linear_data(x -> 3x + 5)
        X_train, y_train, X_test, y_test = split_train_test(X, y; test_size=0.2, shuffle=false)
        
        model = Flux.Dense(1 => 1)
        trainer = Trainer(:linear_regression)
        @show train!(model, X_train, y_train, trainer)
        
        w = model.weight
        b = model.bias
        Plots.plot(vec(X), vec(y), seriestype = :scatter, label="True values", title="Model Training")
        Plots.plot!((x) -> b[1] + w[1] * x, label="After Training", lw=2)
        Plots.savefig("linear_regression.png")
    end

    @testset "multiple_linear_regression" begin
        X, y = sample_multiple_linear_data(x -> 3x[1] + 2x[2] - x[3] + 4x[4] - 2x[5] + 1)
        X_train, y_train, X_test, y_test = split_train_test(X, y)
        
        model = Flux.Dense(5 => 1)
        trainer = Trainer(:multiple_linear_regression)
        train!(model, X_train, y_train, trainer)
        @show train!(model, X_train, y_train, trainer)
    end

    @testset "polynomial_regression" begin
        X, y = sample_polynomial_data(x -> 2 + 3x + 5x^2 - 3x^3)
        X_train, y_train, X_test, y_test = split_train_test(X, y)
        
        model = Flux.Dense(3 => 1)
        trainer = Trainer(:polynomial_regression; learning_rate=0.0003)
        train!(model, X_train, y_train, trainer)
        @show train!(model, X_train, y_train, trainer)

        w = model.weight
        b = model.bias
        Plots.plot(X[1, :], vec(y), seriestype=:scatter, label="True values", title="Polynomial Regression")
        Plots.plot!((x) -> b[1] + w[3] * x^3 + w[2] * x^2 + w[1] * x, label="Predicted values", lw=2)
        Plots.savefig("polynomial_regression.png")
    end
end