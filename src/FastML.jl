module FastML

include("models.jl")
export ModelType, LinearRegression, MultipleLinearRegression, PolynomialRegression, LogisticRegression, NeuralNetworkRegression
export RegType, None, L1, L2, ElasticNet
export Trainer

include("utils.jl")
export split_train_test, r2

include("train.jl")
export train!

end
