module FastML

include("models.jl")
export ModelType, LinearRegression, PolynomialRegression, LogisticRegression, NeuralNetwork
export RegType, None, L1, L2, ElasticNet
export Trainer

include("utils.jl")
export split_train_test

include("train.jl")
export train!

end
