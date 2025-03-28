module FastML

using Random

export split_train_test

@enum RegType None L1 L2 ElasticNet

struct Trainer
    reg_type::RegType
    lambda1::Float64
    lambda2::Float64
    learning_rate::Float64
    max_epochs::Int
    tolerance::Float64
end

function Trainer(
    reg_type::RegType = None,
    lambda1::Float64 = 0.1,
    lambda2::Float64 = 0.1,
    learning_rate::Float64 = 0.01,
    max_epochs::Int = 1000,
    tolerance::Float64 = 1e-6
) 
    return new(reg_type, lambda1, lambda2, learning_rate, max_epochs, tolerance)
end


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

    return X_train, X_test, y_train, y_test
end

end
