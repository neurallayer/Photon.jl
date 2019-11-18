
export  L1Loss, MAELoss, L2Loss, MSELoss, LNLoss, PseudoHuberLoss, HingeLoss,
  BCELoss, CrossEntropyLoss, FocalLoss


"""
Calculates the mean absolute error between `label` and `pred`.

``L = \\sum_i \\vert {label}_i - {pred}_i \\vert``

`label` and `pred` can have arbitrary shape as long as they have the same
number of elements.
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
Calculates the mean squared error between `label` and `pred`.

``L = \\frac{1}{2} \\sum_i ({label}_i - {pred}_i)^2``

`label` and `pred` can have arbitrary shape as long as they have the same
number of elements.
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
More generic version of L1loss and L2loss that allows L{N}Loss.

``L = \\frac{1}{2} \\sum_i \\vert {label}_i - {pred}_i \\vert^{N}``

`label` and `pred` can have arbitrary shape as long as they have the same
number of elements.

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
BCE loss is useful when training logistic regression.

``L = - \\sum_i {label}_i * \\log({pred}_i) +
            (1 - {label}_i) * \\log(1 - {pred}_i)``

If `use_sigmoid` is true, first a sigmoid will be applied before
ccluating the loss.
"""
struct BCELoss <: Loss
  reduction
  use_sigmoid
  BCELoss(;reduction=mean, use_sigmoid=false) = new(reduction, use_sigmoid)
end

function (l::BCELoss)(ŷ, y)
  ŷ, y = Knet.mat(ŷ), Knet.mat(y)

  ŷ = l.use_sigmoid ? sigm(ŷ) : ŷ

  r = -y .* log.(ŷ .+ ϵ) .- (1 .- y) .* log.(1 .- ŷ .+ ϵ)
  l.reduction(r)
end

"""
CrossEntropy loss function with support for an optional weight parameter.
The weight parameter can be static (for example to handle class inbalances)
or dynamic, so passed every time when the lost function is invoked.

Cross Entropy Function:

``L = - \\sum_i {label}_i \\log({output}_i)``

If use_softmax is true, first the softmax(output) will be performed before
cross entropy.

# Usage

```julia
workout = Workout(model, CrossEntropyLoss(), SGD())
```

"""
struct CrossEntropyLoss <: Loss
  weight
  reduction
  use_softmax
  CrossEntropyLoss(weight=1; reduction=mean, use_softmax=false) =
    new(weight, reduction, use_softmax)
end

function (l::CrossEntropyLoss)(ŷ, y, weight=nothing)
  w = weight !== nothing ? weight : l.weight
  ŷ, y = Knet.mat(ŷ), Knet.mat(y)

  ŷ = l.use_softmax ? softmax(ŷ) : ŷ

  r = sum((y .* log.(ŷ .+ ϵ)) .* w, dims=1)
  l.reduction(-r)
end


"""
Calculates the hinge loss function often used in SVMs:

``L = \\sum_i max(0, {margin} - {pred}_i \\cdot {label}_i)``

where `pred` is the classifier prediction and `label` is the target tensor
containing values -1 or 1. `label` and `pred` must have the same number of
elements.

If autofix is true, will convert label from {0,1} to {-1,1}

"""
struct HingeLoss <: Loss
  reduction
  margin
  autofix
  HingeLoss(;margin=1.0, reduction=mean, autofix=true) = new(reduction,margin, autofix)
end

function (l::HingeLoss)(ŷ, y)
  ŷ, y = Knet.mat(ŷ), Knet.mat(y)

  # Hinge lost requires y to be {-1,1}. So the following converts {0,1} to
  # the right values
  if l.autofix
    y[Array(y) .< 1.0] .= -1.0
  end

  r = max.(0, l.margin .- (ŷ .* y))
  l.reduction(r)
end


"""
Calculates the soft-margin loss function used in SVMs:

``L = \\sum_i max(0, {margin} - {pred}_i \\cdot {label}_i)^2``

where `pred` is the classifier prediction and `label` is the target tensor
containing values -1 or 1. `label` and `pred` can have arbitrary shape as
long as they have the same number of elements.

"""
struct SquaredHingeLoss <: Loss
  reduction
  margin
  autofix
  SquaredHingeLoss(;margin=1.0, reduction=mean, autofix=true) = new(reduction,margin, autofix)
end

function (l::SquaredHingeLoss)(ŷ, y)
  ŷ, y = Knet.mat(ŷ), Knet.mat(y)

  # Hinge lost requires y to be {-1,1}. So the following converts {0,1} to
  # the right values
  if l.autofix
    y[Array(y) .< 1.0] .= -1.0
  end

  r = max.(0, l.margin .- (ŷ .* y)).^2
  l.reduction(r)
end



"""
Focal loss for multi-classification

``L=-alpha(1-p_t)^{gamma}ln(p_t)``

See also Focal Loss for Dense Object Detection
    https://arxiv.org/abs/1708.02002
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
