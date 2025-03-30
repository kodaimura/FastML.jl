# ===================== PUBLIC ENUMERATIONS ===================== #
# These enumerations are publicly available and define key model configurations.

""" 
Defines the type of trainer:  
- `Regression`: For regression models  
- `Classification`: For classification models  
"""
@enum TrainerType begin
    Regression
    Classification
end

""" 
Defines the type of model that can be used.  
Includes regression models, logistic regression, and neural networks.  
"""
@enum ModelType begin
    Linear
    MultipleLinear
    Polynomial
    NeuralNetwork
    BinaryLogistic
    SoftmaxLogistic
    SoftmaxNeuralNetwork
end

""" 
Defines the type of regularization used in training:  
- `None`: No regularization  
- `L1`: Lasso regularization  
- `L2`: Ridge regularization  
- `ElasticNet`: Combination of L1 and L2  
"""
@enum RegType begin
    None
    L1
    L2
    ElasticNet
end

# ===================== PUBLIC TRAINER STRUCTS ===================== #
# These trainer structures are externally exposed and used for training models.

""" 
Trainer for regression models.  

Supports different regression models (`Linear`, `MultipleLinear`, `Polynomial`, `NeuralNetwork`)  
and allows various types of regularization.
"""
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

""" 
Trainer for binary classification models (e.g., logistic regression).  

Supports `BinaryLogistic` models with configurable regularization.
"""
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

""" 
Trainer for multi-class classification models (e.g., softmax regression, neural networks).  

Supports `SoftmaxLogistic` and `SoftmaxNeuralNetwork` models with different regularization techniques.
"""
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

# ===================== INTERNAL FUNCTIONS ===================== #
# These functions are used internally for validation.

""" 
Validates if the given model type is allowed for the specified task.  
Throws an assertion error if the model type is invalid.
"""
function _validate_model_type(input::Symbol, allowed_models::Dict{Symbol, ModelType})
    @assert input in keys(allowed_models) 
            "Invalid model_type: $input. Allowed: $(keys(allowed_models))"
end

""" 
Validates if the given regularization type is allowed.  
Throws an assertion error if the regularization type is invalid.
"""
function _validate_reg_type(input::Symbol, allowed_regs::Dict{Symbol, RegType})
    @assert input in keys(allowed_regs) 
            "Invalid reg_type: $input. Allowed: $(keys(allowed_regs))"
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
