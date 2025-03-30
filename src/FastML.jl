module FastML

# ===================== MODELS ===================== #
include("models.jl")

# Exporting trainer types
export RegressorTrainer,
       BinaryClassifierTrainer,
       SoftmaxClassifierTrainer

# Exporting trainer categories
export TrainerType,
       Regression,
       Classification

# Exporting model types
export ModelType, 
       Linear, 
       MultipleLinear, 
       Polynomial, 
       NeuralNetwork,
       BinaryLogistic, 
       SoftmaxLogistic, 
       SoftmaxNeuralNetwork

# Exporting regularization types
export RegType, 
       None, 
       L1, 
       L2, 
       ElasticNet

# ===================== UTILS ===================== #
include("utils.jl")

# Exporting utility functions for data processing
export split_train_test,
       sample_linear_regression_data, 
       sample_multiple_linear_regression_data, 
       sample_polynomial_regression_data, 
       sample_classification_data,
       sample_binary_classification_data

# ===================== TRAINING ===================== #
include("training.jl")

# Exporting core training and evaluation functions
export train!, 
       r2, 
       accuracy

end
