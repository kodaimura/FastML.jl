using Flux
using Statistics

export LinearRegression, train!, predict, r2, weight, bias

@enum RegType None L1 L2 ElasticNet

mutable struct LinearRegression
    input_size::Int
    reg_type::RegType
    lambda1::Float64
    lambda2::Float64
    learning_rate::Float64
    max_epochs::Int
    tolerance::Float64
    _model::Flux.Dense

    function LinearRegression(input_size::Int = 1;
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

function get_model(lr::LinearRegression)
    return lr._model
end

function predict(lr::LinearRegression, features)
    model = get_model(lr)
    return model(features)
end


function weight(lr::LinearRegression)
    return get_model(lr).weight
end

function bias(lr::LinearRegression)
    return get_model(lr).bias
end

function train!(lr::LinearRegression, data)::Tuple{Bool, Int, Float64}
    features = hcat([d[1] for d in data]...)
    labels = hcat([d[2] for d in data]...)
    train!(lr, features, labels)
end

function train!(lr::LinearRegression, features, labels)::Tuple{Bool, Int, Float64}
    model = get_model(lr)
    loss = create_loss_function(lr)
    prev_loss = Inf
    epoch_loss = 0.0
    for epoch in 1:lr.max_epochs
        train_model!(loss, model, features, labels; learning_rate=lr.learning_rate)
        epoch_loss = loss(model, features, labels)

        if abs(prev_loss - epoch_loss) < lr.tolerance
            return true, epoch, epoch_loss
        end
        prev_loss = epoch_loss
    end
    return false, lr.max_epochs, epoch_loss
end

function create_loss_function(lr::LinearRegression)
    reg_type = lr.reg_type
    if reg_type == L1
        return (model::Flux.Dense, x, y) -> loss_lasso(model, x, y, lr.lambda1)
    elseif reg_type == L2
        return (model::Flux.Dense, x, y) -> loss_ridge(model, x, y, lr.lambda2)
    elseif reg_type == ElasticNet
        return (model::Flux.Dense, x, y) -> loss_elastic_net(model, x, y, lr.lambda1, lr.lambda2)
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

function r2(lr::LinearRegression, features, labels)
    y = labels
    y_pred = predict(lr, features)

    ss_tot = sum((y .- mean(y)) .^ 2)
    ss_res = sum((y .- y_pred) .^ 2)

    return 1 - ss_res / ss_tot
end