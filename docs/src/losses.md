

## Loss functions
Loss functions can be used both to calculate the loss as the last step in a
neural network as well as calculate metrics. In either cases they are provided
as an argument to the Workout constructor

```julia
# L1Loss as a loss
workout = Workout(model, L1Loss())

# BCELoss as a loss function and FocalLoss as a metric
workout = Workout(model, BCELoss(), floss=FocalLoss())
```


```@docs
Loss
L1Loss
L2Loss
LNLoss
PseudoHuberLoss
BCELoss
CrossEntropyLoss
HingeLoss
SquaredHingeLoss
FocalLoss
```
