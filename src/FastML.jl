module FastML

include("linear_regression.jl")
include("polynomial_regression.jl")
include("logistic_regression.jl")

using .LinearRegression
using .PolynomialRegression
using .LogisticRegression
using Random

export Model, train!, predict, r2, weight, bias
export split_data

function split_data(x, y, test_size::Float64)
    n = size(x, 2)
    indices = randperm(n)
    test_size = round(Int, test_size * n)
    
    test_indices = indices[1:test_size]
    train_indices = indices[test_size+1:end]

    x_train = x[:, train_indices]
    y_train = y[:, train_indices]
    x_test = x[:, test_indices]
    y_test = y[:, test_indices]
    
    return x_train, y_train, x_test, y_test
end

end
