using Test
using Flux
using FastML
using Plots
using Random
using Statistics

@testset "train!" begin
    @testset "linear_regression" begin
        X, y = sample_linear_data(x -> 3x + 5)
        X_train, y_train, X_test, y_test = split_train_test(X, y)
        
        model = Dense(1 => 1)
        trainer = Trainer(:linear_regression)
        @show train!(model, X_train, y_train, trainer)
        
        #w = model.weight
        #b = model.bias
        #Plots.plot(vec(X), vec(y), seriestype = :scatter, label="True values", title="Model Training")
        #Plots.plot!((x) -> b[1] + w[1] * x, label="After Training", lw=2)
        #Plots.savefig("linear_regression.png")
    end

    @testset "multiple_linear_regression" begin
        X, y = sample_multiple_linear_data(x -> 3x[1] + 2x[2] - x[3] + 4x[4] - 2x[5] + 1)
        X_train, y_train, X_test, y_test = split_train_test(X, y)
        
        model = Dense(5 => 1)
        trainer = Trainer(:multiple_linear_regression)
        @show train!(model, X_train, y_train, trainer)
    end

    @testset "polynomial_regression" begin
        X, y = sample_polynomial_data(x -> 2 + 3x + 5x^2 - 3x^3)
        X_train, y_train, X_test, y_test = split_train_test(X, y)
        
        model = Dense(3 => 1)
        trainer = Trainer(:polynomial_regression; learning_rate=0.0003)
        @show train!(model, X_train, y_train, trainer)

        #w = model.weight
        #b = model.bias
        #Plots.plot(X[1, :], vec(y), seriestype=:scatter, label="True values", title="Polynomial Regression")
        #Plots.plot!((x) -> b[1] + w[3] * x^3 + w[2] * x^2 + w[1] * x, label="Predicted values", lw=2)
        #Plots.savefig("polynomial_regression.png")
    end

    @testset "neural_network_regression" begin
        X, y = sample_multiple_linear_data(x -> 3x[1] + 2x[2] - x[3] + 4x[4] - 2x[5] + 1)
        X_train, y_train, X_test, y_test = split_train_test(X, y)
        
        model = Chain(Dense(5 => 20, σ), Dense(20 => 1))
        trainer = Trainer(:neural_network_regression; learning_rate=0.01, max_epochs=1000)
        @show train!(model, X_train, y_train, trainer)
    end

    @testset "logistic_regression" begin
        classes = [1, 2, 3];
        X, y = sample_classification_data(3, 2; samples=100, x_min=-10, x_max=10)
        #scatter(X[1, :], X[2, :], c=y[:], legend=false, markersize=3)
        #Plots.savefig("classes.png")

        X_train, y_train, X_test, y_test = split_train_test(X, y; shuffle=true)
        model = Chain(Dense(2 => 3), softmax)
        trainer = Trainer(:logistic_regression; learning_rate=0.05, max_epochs=10000)
        @show train!(model, X_train, y_train, classes, trainer)

        @show accuracy(model, X_test, y_test, classes)

        #x_min, x_max = minimum(X[1, :]) - 1, maximum(X[1, :]) + 1
        #y_min, y_max = minimum(X[2, :]) - 1, maximum(X[2, :]) + 1
        #xx = range(x_min, stop=x_max, length=100)
        #yy = range(y_min, stop=y_max, length=100)
        #zz = [Flux.onecold(model([x, y]), classes)[1] for x in xx, y in yy]
        #scatter(X[1, :], X[2, :], c=y[:], marker=:circle, label="Training Data")
        #contour!(xx, yy, zz', levels=length(classes), linewidth=2, color=:black, label="Decision Boundary")
        #Plots.savefig("logistic_regression.png")
    end
end

@testset "r2" begin
    X, y = sample_linear_data(x -> 3x + 5)
    X_train, y_train, X_test, y_test = split_train_test(X, y; test_size=0.2, shuffle=false)

    model = Dense(1 => 1)
    trainer = Trainer(:linear_regression)
    train!(model, X_train, y_train, trainer)

    #@show r2(model, X_train, y_train), r2(model, X_test, y_test)

    @test 1 > r2(model, X_train, y_train) > 0.8
    @test 1 > r2(model, X_test, y_test) > 0.8
end