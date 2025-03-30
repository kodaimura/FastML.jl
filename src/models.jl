@enum TrainerType begin
    Regression
    Classification
end

@enum ModelType begin
    Linear
    MultipleLinear
    Polynomial
    NeuralNetwork
    BinaryLogistic
    SoftmaxLogistic
    SoftmaxNeuralNetwork
end

@enum RegType begin
    None
    L1
    L2
    ElasticNet
end

const ALLOWED_MODELS = Dict(
    :regression => Dict(
        :linear => Linear,
        :multiple_linear => MultipleLinear,
        :polynomial => Polynomial,
        :neural_network => NeuralNetwork,
    ),
    :binary_classification => Dict(
        :logistic => BinaryLogistic,
    ),
    :softmax_classification => Dict(
        :logistic => SoftmaxLogistic,
        :neural_network => SoftmaxNeuralNetwork,
    ),
)

const ALLOWED_REGULARIZATIONS = Dict(
    :none => None,
    :l1 => L1,
    :l2 => L2,
    :elastic_net => ElasticNet
)

""" 回帰モデル用のトレーナー """
struct RegressorTrainer
    trainer_type::TrainerType
    model_type::ModelType
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
        _validate_model_type(model_type, ALLOWED_MODELS[:regression])
        _validate_reg_type(reg_type, ALLOWED_REGULARIZATIONS)

        return new(
            Regression,
            ALLOWED_MODELS[:regression][model_type],
            ALLOWED_REGULARIZATIONS[reg_type], 
            lambda1, lambda2, learning_rate, max_epochs, tolerance
        )
    end
end

""" バイナリ分類（ロジスティック回帰など）用のトレーナー """
struct BinaryClassifierTrainer
    trainer_type::TrainerType
    model_type::ModelType
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
        _validate_model_type(model_type, ALLOWED_MODELS[:binary_classification])
        _validate_reg_type(reg_type, ALLOWED_REGULARIZATIONS)

        return new(
            Classification,
            ALLOWED_MODELS[:binary_classification][model_type],
            ALLOWED_REGULARIZATIONS[reg_type], 
            lambda1, lambda2, learning_rate, max_epochs, tolerance
        )
    end
end

""" 多クラス分類（Softmax, NN分類）用のトレーナー """
struct SoftmaxClassifierTrainer
    trainer_type::TrainerType
    model_type::ModelType
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
        _validate_model_type(model_type, ALLOWED_MODELS[:softmax_classification])
        _validate_reg_type(reg_type, ALLOWED_REGULARIZATIONS)

        return new(
            Classification,
            ALLOWED_MODELS[:softmax_classification][model_type],
            ALLOWED_REGULARIZATIONS[reg_type], 
            lambda1, lambda2, learning_rate, max_epochs, tolerance
        )
    end
end

function _validate_model_type(input::Symbol, allowed_models::Dict{Symbol, ModelType})
    @assert input in keys(allowed_models) 
            "Invalid model_type: $input. Allowed: $(keys(allowed_models))"
end

function _validate_reg_type(input::Symbol, allowed_regs::Dict{Symbol, RegType})
    @assert input in keys(allowed_regs) 
            "Invalid reg_type: $input. Allowed: $(keys(allowed_regs))"
end
