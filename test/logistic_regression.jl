using Test
using FastML
using Random
using Plots

gr()

@testset "logistic_regression" begin
    Random.seed!(42)
    x = randn(1000, 5)
    y = Int[]
    for i in 1:1000
        feature1, feature2, feature3, feature4, feature5 = x[i, :]
        rule_sum = feature1 + feature2 * 2 - feature3 * 1.5 + feature4 * 0.5 - feature5 * 0.8
        rand_factor = rand()

        if rule_sum > 3.0 && rand_factor > 0.3
            push!(y, 1)
        elseif rule_sum > 0.5 && rand_factor > 0.6
            push!(y, 2)
        else
            push!(y, 3)
        end
    end

    x = x'
    y = y'
    x_train, y_train, x_test, y_test = split_data(x, y, 0.2)

    y_train = y_train'
    y_test = y_test'
    classes = [1, 2, 3]
    model = FastML.LogisticRegression.Model(5, 3;learning_rate=0.0003,max_epochs=10000,tolerance=1e-6)
    println(FastML.LogisticRegression.train!(model, x_train, y_train, classes))

    #@test 1 > FastML.LogisticRegression.r2(model, x_train, y_train) > 0.6
    #@test 1 > FastML.LogisticRegression.r2(model, x_test, y_test) > 0.6

    #w = FastML.LogisticRegression.weight(model)
    #b = FastML.LogisticRegression.bias(model)
end