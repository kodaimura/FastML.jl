@enum ModelType begin
    LinearRegression
    MultipleLinearRegression
    PolynomialRegression
    LogisticRegression
    NeuralNetworkRegression
end

@enum RegType begin
    None
    L1
    L2
    ElasticNet
end

const ALLOWED_MODELS = Dict(
    :linear_regression => LinearRegression,
    :multiple_linear_regression => MultipleLinearRegression,
    :polynomial_regression => PolynomialRegression,
    :logistic_regression => LogisticRegression,
    :neural_network_regression => NeuralNetworkRegression
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