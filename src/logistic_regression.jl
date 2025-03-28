module LogisticRegression

using Flux
using Statistics

export Model, train!, predict, r2, weight, bias

@enum RegType None L1 L2 ElasticNet

mutable struct Model
    input_size::Int
    output_size::Int
    reg_type::RegType
    lambda1::Float64
    lambda2::Float64
    learning_rate::Float64
    max_epochs::Int
    tolerance::Float64
    _model::Flux.Chain

    function Model(input_size::Int = 1, output_size::Int = 1;
        reg_type::RegType = None,
        lambda1::Float64 = 0.0,
        lambda2::Float64 = 0.0,
        learning_rate::Float64 = 0.1,
        max_epochs::Int = 1000,
        tolerance::Float64 = 1e-5
    )
    
        return new(
            input_size,
            output_size,
            reg_type,
            lambda1, 
            lambda2, 
            learning_rate,
            max_epochs,
            tolerance,
            Flux.Chain(Flux.Dense(input_size => output_size), softmax)
        )
    end
end

function get_model(lrm::Model)
    return lrm._model
end

function predict(lrm::Model, features)
    model = get_model(lrm)
    return model(features)
end


function weight(lrm::Model)
    return get_model(lrm).weight
end

function bias(lrm::Model)
    return get_model(lrm).bias
end

function train!(lrm::Model, data, classes)::Tuple{Bool, Int, Float64}
    features = reduce(hcat, first.(data))
    labels = reduce(hcat, last.(data))
    train!(lrm, features, labels, classes)
end

function train!(lrm::Model, features, labels, classes)::Tuple{Bool, Int, Float64}
    labels_onehot = Flux.onehotbatch(labels, classes)
    println(size(labels))
    println(size(labels_onehot))
    model = get_model(lrm)
    loss = create_loss_function(lrm)
    prev_loss = Inf
    epoch_loss = 0.0
    for epoch in 1:lrm.max_epochs
        train_model!(loss, model, features, labels_onehot; learning_rate=lrm.learning_rate)
        epoch_loss = loss(model, features, labels_onehot)

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
        return (model::Flux.Chain, x, y) -> loss_lasso(model, x, y, lrm.lambda1)
    elseif reg_type == L2
        return (model::Flux.Chain, x, y) -> loss_ridge(model, x, y, lrm.lambda2)
    elseif reg_type == ElasticNet
        return (model::Flux.Chain, x, y) -> loss_elastic_net(model, x, y, lrm.lambda1, lrm.lambda2)
    else
        return (model::Flux.Chain, x, y) -> loss_logitcrossentropy(model, x, y)
    end
end

function loss_base(model::Flux.Chain, features, labels_onehot; lambda1=0.0, lambda2=0.0)
    y_hat = model(features)
    loss = Flux.logitcrossentropy(y_hat, labels_onehot)
    l1_penalty = 0#lambda1 * sum(abs.(model[1].weight))
    l2_penalty = 0#lambda2 * sum(model[1].weight .^ 2)
    return loss + l1_penalty + l2_penalty
end

function loss_logitcrossentropy(model::Flux.Chain, features, labels_onehot)
    return loss_base(model, features, labels_onehot)
end

function loss_lasso(model::Flux.Chain, features, labels_onehot; lambda1=0.0)
    return loss_base(model, features, labels_onehot, lambda1=lambda1)
end

function loss_ridge(model::Flux.Chain, features, labels_onehot; lambda2=0.0)
    return loss_base(model, features, labels_onehot; lambda2=lambda2)
end

function loss_elastic_net(model::Flux.Chain, features, labels_onehot; lambda1=0.0, lambda2=0.0)
    return loss_base(model, features, labels_onehot; lambda1=lambda1, lambda2=lambda2)
end

function train_model!(loss, model::Flux.Chain, features, labels_onehot; learning_rate=0.01)
    data = [(features[:, i], labels_onehot[i]) for i in 1:size(features, 2)]
    train_model!(loss, model, data; learning_rate=learning_rate)
end

function train_model!(loss, model::Flux.Chain, data; learning_rate=0.01)
    Flux.train!(loss, model, data, Descent(learning_rate))
end

end