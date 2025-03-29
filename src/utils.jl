using Random
using Statistics


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

function r2(model, X, y)
    y_pred = model(X)
    y_mean = mean(y[:])

    ss_tot = sum((y .- y_mean) .^ 2)
    ss_res = sum((y .- y_pred) .^ 2)

    return ss_tot == 0 ? 1.0 : 1 - ss_res / ss_tot
end