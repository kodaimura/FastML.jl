module FastML

include("models.jl")
export ModelType, 
       LinearRegression, 
       MultipleLinearRegression, 
       PolynomialRegression, 
       LogisticRegression, 
       NeuralNetworkRegression,
       RegType, None, L1, L2, ElasticNet,
       Trainer

include("utils.jl")
export split_train_test,
       sample_linear_regression_data, 
       sample_multiple_linear_regression_data, 
       sample_polynomial_regression_data, 
       sample_classification_data

include("train.jl")
export train!, r2, accuracy

end
