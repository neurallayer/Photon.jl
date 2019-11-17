
export  L1Loss, MAELoss, L2Loss, MSELoss, LNLoss, PseudoHuberLoss, HingeLoss,
  BCELoss, CrossEntropyLoss, FocalLoss


binarycrossentropy(ŷ, y) = -y*log(ŷ + ϵ) - (1 - y)*log(1 - ŷ + ϵ)


function bce_loss(y_pred, y)
    y_pred = Knet.mat(y_pred)
    y = Knet.mat(y_pred)
    # y_pred = Knet.sigm.(y_pred)
    loss = binarycrossentropy.(y_pred, y)
    mean(loss)
end


"""
L1Loss also known as MAE loss
"""
struct L1Loss <: Loss
  reduction

  L1Loss(;reduction=mean) = new(reduction)
end

function (l::L1Loss)(ŷ, y)
  ŷ, y = Knet.mat(ŷ), Knet.mat(y)
  l.reduction(abs.(ŷ - y))
end


MAELoss = L1Loss

"""
L2Loss also known as MSE loss
"""
struct L2Loss <: Loss
  reduction

  L2Loss(;reduction=mean) = new(reduction)
end

function (l::L2Loss)(ŷ, y)
  ŷ, y = Knet.mat(ŷ), Knet.mat(y)
  l.reduction((ŷ - y).^2)
end

MSELoss = L2Loss

"""
LNLoss
"""
struct LNLoss <: Loss
  n
  reduction

  LNLoss(n::Int;reduction=mean) = new(n, reduction)
end

function (l::LNLoss)(ŷ, y)
  ŷ, y = Knet.mat(ŷ), Knet.mat(y)
  l.reduction(abs.(ŷ - y).^l.n)
end

"""
Pseudo Huber Loss implementation, somewhere between a L1 and L2 loss.
"""
struct PseudoHuberLoss <: Loss
  delta
  reduction
  PseudoHuberLoss(delta=1; reduction=mean) = new(delta, reduction)
end

function (l::PseudoHuberLoss)(ŷ, y)
  ŷ, y = Knet.mat(ŷ), Knet.mat(y)
  δ = l.delta
  r = δ^2 .* (sqrt.(1 .+ ((ŷ - y)/δ).^2) .- 1)
  l.reduction(r)
end


"""
Binary CrossEntropy
"""
struct BCELoss <: Loss
  reduction

  BCELoss(;reduction=mean) = new(reduction)
end

function (l::BCELoss)(ŷ, y)
  ŷ, y = Knet.mat(ŷ), Knet.mat(y)
  r = -y .* log.(ŷ .+ ϵ) .- (1 .- y) .* log.(1 .- ŷ .+ ϵ)
  l.reduction(r)
end

"""
CrossEntropy loss function with support for an optional weight parameter.
The weight parameter can be static (for example to handle class inbalances)
or dynamic, so passed every time when the lost function is invoked.

# Usage

```julia
workout = Workout(model, CrossEntropyLoss(), SGD())
```

"""
struct CrossEntropyLoss <: Loss
  weight
  reduction
  CrossEntropyLoss(weight=1; reduction=mean) = new(weight, reduction)
end

function (l::CrossEntropyLoss)(ŷ, y, weight=nothing)
  w = weight !== nothing ? weight : l.weight

  ŷ, y = Knet.mat(ŷ), Knet.mat(y)
  r = sum((y .* log.(ŷ .+ ϵ)) .* w, dims=1)
  l.reduction(-r)
end



"""
Hinge Loss implementation
"""
struct HingeLoss <: Loss
  reduction
  autofix
  HingeLoss(;reduction=mean, autofix=true) = new(reduction,autofix)
end

function (l::HingeLoss)(ŷ, y)
  ŷ, y = Knet.mat(ŷ), Knet.mat(y)

  # Hinge lost requires y to be {-1,1}. So the following converts {0,1} to
  # the right values
  if l.autofix
    y[Array(y) .< 1.0] .= -1.0
  end

  r = max.(0, 1 .- (ŷ .* y))
  l.reduction(r)
end


"""
Focal Loss implementation
"""
struct FocalLoss <: Loss
  reduction
  gamma
  alpha
  FocalLoss(;gamma=2, alpha=2, reduction=mean) = new(reduction, gamma, alpha)
end

function (l::FocalLoss)(ŷ, y)
  ŷ, y = Knet.mat(ŷ), Knet.mat(y)
  γ = l.gamma

  loss =   @.      y  * ( - (1 - ŷ)^γ  * log(    ŷ + ϵ)) # if y = 1
  loss .+= @. (1 - y) * ( - (    ŷ)^γ  * log(1 - ŷ + ϵ)) # if y = 0

  l.reduction(loss)
end
