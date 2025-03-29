module FastML

using Random

export ModelType, LinearRegression, PolynomialRegression, LogisticRegression, NeuralNetwork
export RegType, None, L1, L2, ElasticNet
export Trainer
export split_train_test


@enum ModelType begin
    LinearRegression
    PolynomialRegression
    LogisticRegression
    NeuralNetwork
end

@enum RegType begin
    None
    L1
    L2
    ElasticNet
end

const ALLOWED_MODELS = Dict(
    :linear_regression => LinearRegression,
    :polynomial_regression => PolynomialRegression,
    :logistic_regression => LogisticRegression,
    :neural_network => NeuralNetwork
)

const ALLOWED_REGULARIZATIONS = Dict(
    :none => None,
    :l1 => L1,
    :l2 => L2,
    :elastic_net => ElasticNet
)

struct Trainer
    model_type::ModelType
    reg_type::RegType
    lambda1::Float64
    lambda2::Float64
    learning_rate::Float64
    max_epochs::Int
    tolerance::Float64

    function Trainer(model_type::Symbol, reg_type::Symbol;
        lambda1=0.0, 
        lambda2=0.0, 
        learning_rate=0.01, 
        max_epochs=1000, 
        tolerance=1e-6
    )
        @assert model_type in keys(ALLOWED_MODELS) "Invalid model_type: $model_type"
        @assert reg_type in keys(ALLOWED_REGULARIZATIONS) "Invalid reg_type: $reg_type"

        return new(ALLOWED_MODELS[model_type], ALLOWED_REGULARIZATIONS[reg_type], 
            lambda1, lambda2, learning_rate, max_epochs, tolerance
        )
    end

    function Trainer(model_type::Symbol;
        lambda1=0.0, 
        lambda2=0.0, 
        learning_rate=0.01, 
        max_epochs=1000, 
        tolerance=1e-6
    )
        @assert model_type in keys(ALLOWED_MODELS) "Invalid model_type: $model_type"

        return new(ALLOWED_MODELS[model_type], ALLOWED_REGULARIZATIONS[:none],
            lambda1, lambda2, learning_rate, max_epochs, tolerance
        )
    end
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
