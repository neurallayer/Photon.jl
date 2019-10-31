
export mse, mae, bce_loss, MSELoss, PseudoHuberLoss, MAELoss, HingeLoss,
  BCELoss, CrossEntropyLoss

mae(y_pred, y) = mean(abs.(Knet.mat(y_pred) .- Knet.mat(y)))

mse(y_pred, y) = mean((Knet.mat(y_pred) .- Knet.mat(y)).^2)


binarycrossentropy(ŷ, y) = -y*log(ŷ + ϵ) - (1 - y)*log(1 - ŷ + ϵ)


function bce_loss(y_pred, y)
    y_pred = Knet.mat(y_pred)
    y = Knet.mat(y_pred)
    # y_pred = Knet.sigm.(y_pred)
    loss = binarycrossentropy.(y_pred, y)
    mean(loss)
end

"""
Base type for the loss functions. However Photon accepts
any functoon as a loss function as long as it is callable:

  fn(y_pred, y_true)

and returns the loss as a scalar value.
"""
abstract type Loss end

"""
Mean Square Error implementation, also referred to
as the L2 Loss.

Examples:

  workout = Workout(model, MSELoss(), SGD())
"""
struct MSELoss <: Loss
  reduction::Symbol

  MSELoss(;reduction=:mean) = new(reduction)
end

function (l::MSELoss)(ŷ, y)
  ŷ, y = Knet.mat(ŷ), Knet.mat(y)
  r = sum((ŷ .- y).^2)
  l.reduction == :mean ? r * 1 // length(y) : r
end


"""
Mean Absolute Error implementation, also referred to
as the L1 Loss.
"""
struct MAELoss <: Loss
  reduction::Symbol

  MAELoss(;reduction=:mean) = new(reduction)
end

function (l::MAELoss)(ŷ, y)
  ŷ, y = Knet.mat(ŷ), Knet.mat(y)
  r = sum(abs.(ŷ .- y))
  l.reduction == :mean ? r * 1 // length(y) : r
end

"""
Binary CrossEntropy
"""
struct BCELoss <: Loss
  reduction::Symbol

  BCELoss(;reduction=:mean) = new(reduction)
end

function (l::BCELoss)(ŷ, y)
  ŷ, y = Knet.mat(ŷ), Knet.mat(y)
  r = -y .* log.(ŷ .+ ϵ) .- (1 .- y) .* log.(1 .- ŷ .+ ϵ)
  r = sum(r)
  l.reduction == :mean ? r * 1 // size(y,2) : r
end

"""
CrossEntropy loss function with support for an optional weight parameter.
The weight parameter can be static (for example to handle class inbalances)
or dynamic (so passed every time when the lost function is invoked)

Examples:
  workout = Workout(model, CE(), SGD())

"""
struct CrossEntropyLoss <: Loss
  weight
  reduction::Symbol
  CrossEntropyLoss(;weight=1, reduction=:mean) = new(weight, reduction)
end

function (l::CrossEntropyLoss)(ŷ, y, weight=nothing)
  w = weight != nothing ? weight : l.weight

  ŷ, y = Knet.mat(ŷ), Knet.mat(y)
  r = -sum(y .* log.(ŷ)) .* w
  l.reduction == :mean ? r * 1 // size(y, 2) : r
end

"""
Pseudo Huber Loss implementation, somewhere between a L1 and L2 loss.
"""
struct PseudoHuberLoss <: Loss
  δ
  reduction::Symbol
  PseudoHuberLoss(;δ=1, reduction=:mean) = new(δ, reduction)
end

function (l::PseudoHuberLoss)(ŷ, y)
  δ = l.δ
  r = δ^2 .* (sqrt.(1 .+ ((ŷ - y)/δ).^2) .- 1)
  r = -sum(r)
  l.reduction == :mean ? r * 1 // length(y) : r
end

"""
Hinge Loss implementation
"""
struct HingeLoss <: Loss
  reduction::Symbol
  HingeLoss(;reduction=:mean) = new(reduction)
end

function (l::HingeLoss)(ŷ, y)
  ŷ, y = Knet.mat(ŷ), Knet.mat(y)
  r = max.(0, 1 .- ŷ*y)
  r = -sum(r)
  l.reduction == :mean ? r * 1 // length(y) : r
end
