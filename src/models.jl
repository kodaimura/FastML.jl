@enum RegressionType begin
    Linear
    MultipleLinear
    Polynomial
    NeuralNetwork
end

@enum BinaryClassificationType begin
    BinaryLogistic
end

@enum SoftmaxClassificationType begin
    SoftmaxLogistic
    SoftmaxNeuralNetwork
end

@enum RegType begin
    None
    L1
    L2
    ElasticNet
end

const ALLOWED_REGRESSION_MODELS = Dict(
    :linear => Linear,
    :multiple_linear => MultipleLinear,
    :polynomial => Polynomial,
    :neural_network => NeuralNetwork
)

const ALLOWED_BINARY_CLASSIFICATION_MODELS = Dict(
    :logistic => BinaryLogistic
)

const ALLOWED_SOFTMAX_CLASSIFICATION_MODELS = Dict(
    :logistic => SoftmaxLogistic,
    :neural_network => SoftmaxNeuralNetwork
)

const ALLOWED_REGULARIZATIONS = Dict(
    :none => None,
    :l1 => L1,
    :l2 => L2,
    :elastic_net => ElasticNet
)

""" 回帰モデル用のトレーナー """
struct RegressorTrainer
    model_type::RegressionType
    reg_type::RegType
    lambda1::Float64
    lambda2::Float64
    learning_rate::Float64
    max_epochs::Int
    tolerance::Float64

    function RegressorTrainer(model_type::Symbol, reg_type::Symbol=:none;
        lambda1=0.0, 
        lambda2=0.0, 
        learning_rate=0.01, 
        max_epochs=1000, 
        tolerance=1e-6
    )
        @assert model_type in keys(ALLOWED_REGRESSION_MODELS) "Invalid model_type: $model_type. Allowed: $(keys(ALLOWED_REGRESSION_MODELS))"
        @assert reg_type in keys(ALLOWED_REGULARIZATIONS) "Invalid reg_type: $reg_type. Allowed: $(keys(ALLOWED_REGULARIZATIONS))"

        return new(
            ALLOWED_REGRESSION_MODELS[model_type],
            ALLOWED_REGULARIZATIONS[reg_type], 
            lambda1, lambda2, learning_rate, max_epochs, tolerance
        )
    end
end

""" バイナリ分類（ロジスティック回帰など）用のトレーナー """
struct BinaryClassifierTrainer
    model_type::BinaryClassificationType
    reg_type::RegType
    lambda1::Float64
    lambda2::Float64
    learning_rate::Float64
    max_epochs::Int
    tolerance::Float64

    function BinaryClassifierTrainer(model_type::Symbol, reg_type::Symbol=:none;
        lambda1=0.0, 
        lambda2=0.0, 
        learning_rate=0.01, 
        max_epochs=1000, 
        tolerance=1e-6
    )
        @assert model_type in keys(ALLOWED_BINARY_CLASSIFICATION_MODELS) "Invalid model_type: $model_type. Allowed: $(keys(ALLOWED_BINARY_CLASSIFICATION_MODELS))"
        @assert reg_type in keys(ALLOWED_REGULARIZATIONS) "Invalid reg_type: $reg_type. Allowed: $(keys(ALLOWED_REGULARIZATIONS))"

        return new(
            ALLOWED_BINARY_CLASSIFICATION_MODELS[model_type],
            ALLOWED_REGULARIZATIONS[reg_type], 
            lambda1, lambda2, learning_rate, max_epochs, tolerance
        )
    end
end

""" 多クラス分類（Softmax, NN分類）用のトレーナー """
struct SoftmaxClassifierTrainer
    model_type::SoftmaxClassificationType
    reg_type::RegType
    lambda1::Float64
    lambda2::Float64
    learning_rate::Float64
    max_epochs::Int
    tolerance::Float64

    function SoftmaxClassifierTrainer(model_type::Symbol, reg_type::Symbol=:none;
        lambda1=0.0, 
        lambda2=0.0, 
        learning_rate=0.01, 
        max_epochs=1000, 
        tolerance=1e-6
    )
        @assert model_type in keys(ALLOWED_SOFTMAX_CLASSIFICATION_MODELS) "Invalid model_type: $model_type. Allowed: $(keys(ALLOWED_SOFTMAX_CLASSIFICATION_MODELS))"
        @assert reg_type in keys(ALLOWED_REGULARIZATIONS) "Invalid reg_type: $reg_type. Allowed: $(keys(ALLOWED_REGULARIZATIONS))"

        return new(
            ALLOWED_SOFTMAX_CLASSIFICATION_MODELS[model_type],
            ALLOWED_REGULARIZATIONS[reg_type], 
            lambda1, lambda2, learning_rate, max_epochs, tolerance
        )
    end
end
