using Random
using LinearAlgebra
using Distributions

function split_train_test(X, y; test_size=0.2, shuffle=true, seed=nothing)
    n = size(X, 2)
    n_test = round(Int, n * test_size)
    n_train = n - n_test

    indices = collect(1:n)
    if shuffle
        seed !== nothing && Random.seed!(seed)
        shuffle!(indices)
    end

    train_idx, test_idx = indices[1:n_train], indices[n_train+1:end]

    X_train, X_test = X[:, train_idx], X[:, test_idx]
    y_train, y_test = y[:, train_idx], y[:, test_idx]

    return X_train, y_train, X_test, y_test
end

function sample_linear_regression_data(f::Function = x -> x; n_samples = 100, x_min = -3, x_max = 3)
    X = rand(Float32, n_samples, 1) * (x_max - x_min) .+ x_min
    y = f.(X) .+ rand(Float32, n_samples)
    
    X = reshape(X, n_samples, 1)'
    y = reshape(y, n_samples, 1)'
    
    return X, y
end

function sample_multiple_linear_regression_data(f::Function = x -> x[1] + x[2]; n_samples = 100, x_min = -3, x_max = 3)
    dim = dimension(f)

    X = rand(Float32, n_samples, dim) * (x_max - x_min) .+ x_min
    y = [f(x) for x in eachrow(X)] .+ rand(Float32, n_samples)

    X = reshape(X, n_samples, dim)'
    y = reshape(y, n_samples, 1)'
    
    return X, y
end

function sample_polynomial_regression_data(f::Function = x -> x + x^2; n_samples = 100, x_min = -3, x_max = 3)
    deg = degree(f)

    X = rand(Float32, n_samples, 1) * (x_max - x_min) .+ x_min
    y = f.(X) .+ rand(Float32, n_samples) * deg
    X_poly = [x ^ i for x in vec(X), i in 1:deg]

    X_poly = reshape(X_poly, n_samples, deg)'
    y = reshape(y, n_samples, 1)'

    return X_poly, y
end

function sample_classification_data(classes = [0,1], n_features = 1; n_samples = 100, x_min = -3, x_max = 3)
    n_classes = length(classes)
    centers = [(rand(Float32, n_features) .* (x_max - x_min) .+ x_min) for _ in 1:n_classes]
    
    X = []
    y = []
    for i in 1:n_classes
        mean = centers[i]
        cov = I(n_features) * 0.4
        dist = MvNormal(mean, cov)
        
        n = n_samples ÷ n_classes
        if i == n_classes
            n += n_samples % n_classes
        end

        X_class = rand(dist, n)
        y_class = fill(classes[i], n)
        
        append!(X, eachcol(X_class))
        append!(y, y_class)
    end
    
    X = Float32.(hcat(X...))
    y = reshape(y, 1, n_samples)

    return X, y
end

function sample_binary_classification_data(n_features = 1; n_samples = 100, x_min = -3, x_max = 3)
    sample_classification_data([0,1], n_features; n_samples=n_samples, x_min=x_min, x_max=x_max)
end

function degree(f::Function)
    x_vals = collect(0:60)
    y_vals = f.(x_vals)

    for degree in 1:61
        diff_vals = y_vals
        for _ in 1:degree
            diff_vals = diff(diff_vals)
        end
        
        if all(x -> x == 0, diff_vals)
            return degree - 1
        end
    end
    error("Degree > 60 not supported")
end

function dimension(f::Function)
    for dimension in 1:60
        try
            f(rand(Float64, dimension))
            return dimension 
        catch e
            continue
        end
    end
    error("Dimension > 60 not supported")
end