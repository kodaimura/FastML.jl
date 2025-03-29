using Flux

function train!(model, X, y, t::Trainer)::Tuple{Bool, Int, Float64}
    loss = create_loss_function(t)
    prev_loss = Inf
    epoch_loss = 0.0
    for epoch in 1:t.max_epochs
        train_model!(loss, model, X, y; learning_rate=t.learning_rate)
        epoch_loss = loss(model, X, y)

        if abs(prev_loss - epoch_loss) < t.tolerance
            return true, epoch, epoch_loss
        end
        prev_loss = epoch_loss
    end
    return false, t.max_epochs, epoch_loss
end

function train_model!(loss, model, X, y; learning_rate=0.01)
    data = [(X[:, i], y[i]) for i in 1:size(X, 2)]
    Flux.train!(loss, model, data, Descent(learning_rate))
end

function train_model!(f_loss, model, X, y_onehot, flg::Bool; learning_rate=0.01)
    dLdm, _, _ = gradient(f_loss, model, X, y_onehot)
    @. model[1].weight = model[1].weight - learning_rate * dLdm[:layers][1][:weight]
    @. model[1].bias = model[1].bias - learning_rate * dLdm[:layers][1][:bias]
end

function create_loss_function(t::Trainer)
    model_type = t.model_type
    if model_type == NeuralNetworkRegression
        return (model, X, y) -> loss_mse(model, X, y)
    elseif model_type == LogisticRegression
        return (model, X, y_onehot) -> loss_logitcrossentropy(model, X, y_onehot)
    end

    reg_type = t.reg_type
    if reg_type == L1
        return (model, X, y) -> loss_lasso(model, X, y, t.lambda1)
    elseif reg_type == L2
        return (model, X, y) -> loss_ridge(model, X, y, t.lambda2)
    elseif reg_type == ElasticNet
        return (model, X, y) -> loss_elastic_net(model, X, y, t.lambda1, t.lambda2)
    else
        return (model, X, y) -> loss_mse(model, X, y)
    end
end

function loss_mse(model, X, y)
    y_hat = model(X)
    return  Flux.mse(y_hat, y)
end

function loss_reg(model, X, y; lambda1=0.0, lambda2=0.0)
    loss = loss_mse(model, X, y)
    l1_penalty = lambda1 * sum(abs.(model.weight))
    l2_penalty = lambda2 * sum(model.weight .^ 2)
    return loss + l1_penalty + l2_penalty
end

function loss_lasso(model, X, y; lambda1=0.0)
    return loss_reg(model, X, y, lambda1=lambda1)
end

function loss_ridge(model, X, y; lambda2=0.0)
    return loss_reg(model, X, y; lambda2=lambda2)
end

function loss_elastic_net(model, X, y; lambda1=0.0, lambda2=0.0)
    return loss_reg(model, X, y; lambda1=lambda1, lambda2=lambda2)
end

function loss_logitcrossentropy(model, X, y_onehot)
    y_hat = model(X)
    return Flux.logitcrossentropy(y_hat, y_onehot)
end

function train!(model, X, y, classes, t::Trainer)::Tuple{Bool, Int, Float64}
    y_onehot = reshape(Flux.onehotbatch(y, classes), length(classes), length(y))
    accuracy(x, y) = Statistics.mean(Flux.onecold(model(x), classes) .== y)
    loss = create_loss_function(t)
    epoch_accuracy = 0.0
    for epoch in 1:t.max_epochs
        train_model!(loss, model, X, y_onehot, true; learning_rate=t.learning_rate)
        epoch_accuracy = accuracy(X, reshape(y, :, 1))
        if epoch_accuracy >= 0.95
            println("Converged at epoch $epoch with accuracy $epoch_accuracy")
            return true, epoch, epoch_accuracy
        end
    end
    return false, t.max_epochs, epoch_accuracy
end