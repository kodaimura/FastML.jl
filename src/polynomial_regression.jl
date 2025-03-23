module PolynomialRegression

using Flux
using Statistics

export Model, train!, predict, r2, weight, bias

@enum RegType None L1 L2 ElasticNet

mutable struct Model
    input_size::Int
    reg_type::RegType
    lambda1::Float64
    lambda2::Float64
    learning_rate::Float64
    max_epochs::Int
    tolerance::Float64
    _model::Flux.Dense

    function Model(input_size::Int = 1;
        reg_type::RegType = None,
        lambda1::Float64 = 0.0,
        lambda2::Float64 = 0.0,
        learning_rate::Float64 = 0.1,
        max_epochs::Int = 1000,
        tolerance::Float64 = 1e-5
    )
    
        return new(
            input_size,
            reg_type,
            lambda1, 
            lambda2, 
            learning_rate,
            max_epochs,
            tolerance,
            Flux.Dense(input_size => 1)
        )
    end
end

function get_model(prm::Model)
    return prm._model
end

function predict(prm::Model, features)
    features = transform_features(features, prm.input_size)
    model = get_model(prm)
    return model(features)
end

function weight(prm::Model)
    return get_model(prm).weight
end

function bias(prm::Model)
    return get_model(prm).bias
end

function train!(prm::Model, data)::Tuple{Bool, Int, Float64}
    x = reduce(hcat, first.(data))
    y = reduce(hcat, last.(data))
    train!(prm, features, labels)
end

function train!(prm::Model, features, labels)::Tuple{Bool, Int, Float64}
    features = transform_features(features, prm.input_size)
    model = get_model(prm)
    loss = create_loss_function(prm)
    prev_loss = Inf
    epoch_loss = 0.0
    for epoch in 1:prm.max_epochs
        train_model!(loss, model, features, labels; learning_rate=prm.learning_rate)
        epoch_loss = loss(model, features, labels)

        if abs(prev_loss - epoch_loss) < prm.tolerance
            return true, epoch, epoch_loss
        end
        prev_loss = epoch_loss
    end
    return false, prm.max_epochs, epoch_loss
end

function create_loss_function(prm::Model)
    reg_type = prm.reg_type
    if reg_type == L1
        return (model::Flux.Dense, x, y) -> loss_lasso(model, x, y, prm.lambda1)
    elseif reg_type == L2
        return (model::Flux.Dense, x, y) -> loss_ridge(model, x, y, prm.lambda2)
    elseif reg_type == ElasticNet
        return (model::Flux.Dense, x, y) -> loss_elastic_net(model, x, y, prm.lambda1, prm.lambda2)
    else
        return (model::Flux.Dense, x, y) -> loss_mse(model, x, y)
    end
end

function loss_base(model::Flux.Dense, features, labels; lambda1=0.0, lambda2=0.0)
    y_hat = model(features)
    mse_loss = Flux.mse(y_hat, labels)
    l1_penalty = lambda1 * sum(abs.(model.weight))
    l2_penalty = lambda2 * sum(model.weight .^ 2)
    return mse_loss + l1_penalty + l2_penalty
end

function loss_mse(model::Flux.Dense, features, labels)
    return loss_base(model, features, labels)
end

function loss_lasso(model::Flux.Dense, features, labels; lambda1=0.0)
    return loss_base(model, features, labels, lambda1=lambda1)
end

function loss_ridge(model::Flux.Dense, features, labels; lambda2=0.0)
    return loss_base(model, features, labels; lambda2=lambda2)
end

function loss_elastic_net(model::Flux.Dense, features, labels; lambda1=0.0, lambda2=0.0)
    return loss_base(model, features, labels; lambda1=lambda1, lambda2=lambda2)
end

#function train_model!(loss, model::Flux.Dense, features, labels; learning_rate=0.01)
#    dLdm, _, _ = gradient(loss, model, features, labels)
#    @. model.weight = model.weight - learning_rate * dLdm.weight
#    @. model.bias = model.bias - learning_rate * dLdm.bias
#end

function train_model!(loss, model::Flux.Dense, features, labels; learning_rate=0.01)
    data = [(features[:, i], labels[i]) for i in 1:size(features, 2)]
    train_model!(loss, model, data; learning_rate=learning_rate)
end

function train_model!(loss, model::Flux.Dense, data; learning_rate=0.01)
    Flux.train!(loss, model, data, Descent(learning_rate))
end

function r2(prm::Model, features, labels)
    y = labels
    y_pred = predict(prm, features)

    ss_tot = sum((y .- mean(y)) .^ 2)
    ss_res = sum((y .- y_pred) .^ 2)

    return 1 - ss_res / ss_tot
end

function transform_features(features, power::Int)
    return vcat([features .^ i for i in 1:power]...)
end

end