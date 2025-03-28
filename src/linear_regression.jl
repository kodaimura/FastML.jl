module LinearRegression

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

function get_model(lrm::Model)
    return lrm._model
end

function predict(lrm::Model, features)
    model = get_model(lrm)
    x = transpose(features)
    return model(x)
end

function transpose(x::AbstractMatrix)
    return x'
end

function weight(lrm::Model)
    return get_model(lrm).weight
end

function bias(lrm::Model)
    return get_model(lrm).bias
end

function train!(lrm::Model, features, labels)::Tuple{Bool, Int, Float64}
    x = transpose(features)
    y = transpose(labels)

    # return (done, epoch, loss)
    # done = 学習が完了したかどうか（Bool）
    # epoch = 学習したエポック数（Int）
    # loss = 最終的な損失（Float64）
    return _train!(lrm, x, y)
end

function _train!(lrm::Model, x, y)
    model = get_model(lrm)
    loss = create_loss_function(lrm)
    prev_loss = Inf
    epoch_loss = 0.0
    for epoch in 1:lrm.max_epochs
        train_model!(loss, model, x, y; learning_rate=lrm.learning_rate)
        epoch_loss = loss(model, x, y)

        if abs(prev_loss - epoch_loss) < lrm.tolerance
            return true, epoch, epoch_loss
        end
        prev_loss = epoch_loss
    end
    return false, lrm.max_epochs, epoch_loss
end

function create_loss_function(lrm::Model)
    reg_type = lrm.reg_type
    if reg_type == L1
        return (model::Flux.Dense, x, y) -> loss_lasso(model, x, y, lrm.lambda1)
    elseif reg_type == L2
        return (model::Flux.Dense, x, y) -> loss_ridge(model, x, y, lrm.lambda2)
    elseif reg_type == ElasticNet
        return (model::Flux.Dense, x, y) -> loss_elastic_net(model, x, y, lrm.lambda1, lrm.lambda2)
    else
        return (model::Flux.Dense, x, y) -> loss_mse(model, x, y)
    end
end

function loss_base(model::Flux.Dense, x, y; lambda1=0.0, lambda2=0.0)
    y_hat = model(x)
    mse_loss = Flux.mse(y_hat, y)
    l1_penalty = lambda1 * sum(abs.(model.weight))
    l2_penalty = lambda2 * sum(model.weight .^ 2)
    return mse_loss + l1_penalty + l2_penalty
end

function loss_mse(model::Flux.Dense, x, y)
    return loss_base(model, x, y)
end

function loss_lasso(model::Flux.Dense, x, y; lambda1=0.0)
    return loss_base(model, x, y, lambda1=lambda1)
end

function loss_ridge(model::Flux.Dense, x, y; lambda2=0.0)
    return loss_base(model, x, y; lambda2=lambda2)
end

function loss_elastic_net(model::Flux.Dense, x, y; lambda1=0.0, lambda2=0.0)
    return loss_base(model, x, y; lambda1=lambda1, lambda2=lambda2)
end

function train_model!(loss, model::Flux.Dense, x, y; learning_rate=0.01)
    data = [(x[:, i], y[i]) for i in 1:size(x, 2)]
    train_model!(loss, model, data; learning_rate=learning_rate)
end

function train_model!(loss, model::Flux.Dense, data; learning_rate=0.01)
    Flux.train!(loss, model, data, Descent(learning_rate))
end

function r2(lrm::Model, features, labels)
    y = transpose(labels)
    y_pred = predict(lrm, features)

    ss_tot = sum((y .- mean(y)) .^ 2)
    ss_res = sum((y .- y_pred) .^ 2)

    return 1 - ss_res / ss_tot
end

end