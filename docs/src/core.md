
## Basic

```@docs
Workout
saveWorkout
loadWorkout
validate
predict
fit!
freeze!
unfreeze!
gradients
```

## Context
Context is a core concept in Photon. It determines what device (GPU or CPU) to
use and what datatype (Float64/32/16) is required. Once the context is set,
models and data will use this by default.

```@docs
Context
getContext
setContext
resetContext
autoConvertor
```

## Internal
You normally won't have to invoke the following functions directly when training
a model. But in some cases you might want to write a specialized version of them.

```@docs
Photon.back!
Photon.step!
```
