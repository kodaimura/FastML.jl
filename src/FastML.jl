module FastML

include("models.jl")
export RegressorTrainer,
       BinaryClassifierTrainer,
       SoftmaxClassifierTrainer

export TrainerType,
       Regression,
       Classification

export ModelType, 
       Linear, 
       MultipleLinear, 
       Polynomial, 
       NeuralNetwork,
       BinaryLogistic, 
       SoftmaxLogistic, 
       SoftmaxNeuralNetwork

export RegType, 
       None, 
       L1, 
       L2, 
       ElasticNet

include("utils.jl")
export split_train_test,
       sample_linear_regression_data, 
       sample_multiple_linear_regression_data, 
       sample_polynomial_regression_data, 
       sample_classification_data

include("training.jl")
export train!, r2, accuracy

end
