using Flux

@enum RegType None L1 L2 ElasticNet

struct LinearModel
    model::Flux.Dense
end

struct Trainer
    reg_type::RegType
    lambda1::Float64
    lambda2::Float64
    learning_rate::Float64
    max_epochs::Int
    tolerance::Float64
end

function (lm::LinearModel)(x)
    return lm.model(x)
end

function LinearModel(input_size::Int = 1)
    model = Flux.Dense(input_size => 1)
    return LinearModel(model)
end

function Trainer(;
    reg_type::RegType = None,
    lambda1::Float64 = 0.0,
    lambda2::Float64 = 0.0,
    learning_rate::Float64 = 0.1,
    max_epochs::Int = 1000,
    tolerance::Float64 = 1e-6
)

    return Trainer(
        reg_type,
        lambda1, 
        lambda2, 
        learning_rate,
        max_epochs,
        tolerance
    )
end

function create_loss_function(trainer::Trainer)
    reg_type = trainer.reg_type
    if reg_type == RegType.L1
        return (model::Flux.Dense, x, y) -> loss_lasso(model, x, y, trainer.lambda1)
    elseif reg_type == RegType.L2
        return (model::Flux.Dense, x, y) -> loss_ridge(model, x, y, trainer.lambda2)
    elseif reg_type == RegType.ElasticNet
        return (model::Flux.Dense, x, y) -> loss_elastic_net(model, x, y, trainer.lambda1, trainer.lambda2)
    else
        return (model::Flux.Dense, x, y) -> loss_mse(model, x, y)
    end
end

function train!(trainer::Trainer, lm::LinearModel, data, labels)
    model = lm.model
    loss = create_loss_function(trainer)
    x = hcat([d[1] for d in data]...)
    y = hcat([d[2] for d in data]...)
    loss_prev = Inf
    for epoch in 1:trainer.max_epochs
        train_model!(loss, model, data; learning_rate=trainer.learning_rate)
        current_loss = loss(model, x, y)

        if loss_prev == Inf
            loss_prev = current_loss
            continue
        end

        if current_loss < 1 && abs(loss_prev - current_loss) < trainer.tolerance
            println("Converged at epoch $epoch with loss $current_loss")
            break
        end
        loss_prev = current_loss
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

function train_model!(loss, model::Flux.Dense, features, labels; learning_rate=0.01)
    dLdm, _, _ = gradient(loss, model, features, labels)
    @. model.weight = model.weight - learning_rate * dLdm.weight
    @. model.bias = model.bias - learning_rate * dLdm.bias
end

function train_model!(loss, model::Flux.Dense, data; learning_rate=0.01)
    Flux.train!(loss, model, data, Descent(learning_rate))
end

end