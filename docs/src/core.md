
## Basic

```@docs
Workout
saveWorkout
loadWorkout
validate
predict
train!
freeze!
unfreeze!
stop
gradients
```


## Internal
You normally won't have to invoke the following functions directly when training
a model. But in some cases you might want to write a specialized version of them.

```@docs
Photon.back!
Photon.step!
```
