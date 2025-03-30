using Flux
using Statistics

# ===================== PUBLIC FUNCTIONS ===================== #
# These functions are publicly available for training and evaluation.

"""
Trains a regression model using a `RegressorTrainer`.

# Arguments
- `trainer`: Instance of `RegressorTrainer`.
- `model`: The model to be trained.
- `X, y`: Feature matrix and target values.

# Returns
- `(converged, epochs, final_loss)`: Training status, epochs used, and final loss.
"""
function train!(trainer::RegressorTrainer, model, X, y)::Tuple{Bool, Int, Float64}
    _train!(trainer, model, X, y, build_loss(trainer, loss_mse))
end

"""
Trains a binary classification model using a `BinaryClassifierTrainer`.

# Arguments
- `trainer`: Instance of `BinaryClassifierTrainer`.
- `model`: The model to be trained.
- `X, y`: Feature matrix and binary labels.

# Returns
- `(converged, epochs, final_loss)`: Training status, epochs used, and final loss.
"""
function train!(trainer::BinaryClassifierTrainer, model, X, y)::Tuple{Bool, Int, Float64}
    _train!(trainer, model, X, y, build_loss(trainer, loss_binarycrossentropy))
end

"""
Trains a softmax classification model using a `SoftmaxClassifierTrainer`.

# Arguments
- `trainer`: Instance of `SoftmaxClassifierTrainer`.
- `model`: The model to be trained.
- `X, y`: Feature matrix and class labels.
- `classes`: List of class labels.

# Returns
- `(converged, epochs, final_loss)`: Training status, epochs used, and final loss.
"""
function train!(trainer::SoftmaxClassifierTrainer, model, X, y, classes)::Tuple{Bool, Int, Float64}
    y_onehot = reshape(Flux.onehotbatch(y, classes), length(classes), length(y))
    _train!(trainer, model, X, y_onehot, build_loss(trainer, loss_logitcrossentropy))
end

"""
Computes the R² (coefficient of determination) score.

# Arguments
- `model`: The trained model.
- `X, y`: Feature matrix and target values.

# Returns
- `r2_score`: The R² metric.
"""
function r2(model, X, y)
    y_pred = model(X)
    y_mean = mean(vec(y))

    ss_tot = sum((y .- y_mean) .^ 2)
    ss_res = sum((y .- y_pred) .^ 2)

    return ss_tot == 0 ? 1.0 : 1 - ss_res / ss_tot
end

"""
Computes the accuracy of a classification model.

# Arguments
- `model`: The trained model.
- `X, y`: Feature matrix and class labels.
- `classes`: List of class labels (optional for softmax classification).

# Returns
- `accuracy_score`: The accuracy metric.
"""
function accuracy(model, X, y, classes)
    return mean(Flux.onecold(model(X), classes) .== vec(y)) 
end

"""
Computes the accuracy of a binary classification model.

# Arguments
- `model`: The trained model.
- `X, y`: Feature matrix and binary labels.

# Returns
- `accuracy_score`: The accuracy metric.
"""
function accuracy(model, X, y)
    y_pred = vec(model(X)) .> 0.5
    return mean(Int.(y_pred) .== vec(y))
end

# ===================== INTERNAL FUNCTIONS ===================== #
# These functions are used internally and are NOT publicly exposed.

function _train!(trainer, model, X, y, loss::Function)::Tuple{Bool, Int, Float64}
    data = [(X[:, i], y[:, i]) for i in 1:size(X, 2)]
    opt = Descent(trainer.learning_rate)
    state = Flux.setup(opt, model)

    prev_loss = Inf
    curr_loss = 0.0
    for epoch in 1:trainer.max_epochs
        Flux.train!(loss, model, data, state)
        curr_loss = loss(model, X, y)
        if abs(prev_loss - curr_loss) < trainer.tolerance
            return true, epoch, curr_loss
        end
        prev_loss = curr_loss
    end
    return false, trainer.max_epochs, curr_loss
end

function build_loss(trainer, loss_base::Function)
    reg_type = trainer.reg_type
    lambda1 = trainer.lambda1
    lambda2 = trainer.lambda2

    if reg_type == L1
        return (model, X, y) -> loss_base(model, X, y) + reg_term(model; lambda1=lambda1)
    elseif reg_type == L2
        return (model, X, y) -> loss_base(model, X, y) + reg_term(model; lambda2=lambda2)
    elseif reg_type == ElasticNet
        return (model, X, y) -> loss_base(model, X, y) + reg_term(model; lambda1=lambda1, lambda2=lambda2)
    else
        return (model, X, y) -> loss_base(model, X, y)
    end
end

function loss_mse(model, X, y)
    y_hat = model(X)
    return  Flux.mse(y_hat, y)
end

function loss_binarycrossentropy(model, X, y_onehot)
    y_hat = model(X)
    return Flux.binarycrossentropy(y_hat, y_onehot)
end

function loss_logitcrossentropy(model, X, y_onehot)
    y_hat = model(X)
    return Flux.logitcrossentropy(y_hat, y_onehot)
end

function reg_term(model; lambda1=0.0, lambda2=0.0)
    weights = hasproperty(model, :weight) ? model.weight : model[1].weight
    l1_penalty = lambda1 * sum(abs.(weights))
    l2_penalty = lambda2 * sum(weights .^ 2)
    return  l1_penalty + l2_penalty
end
