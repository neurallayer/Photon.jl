import Base:haslength
import Serialization
import Knet

import Photon.Layers: Layer
import Photon.Losses: Loss


export Workout, saveworkout, loadworkout, predict, train!, hasmetric,
        freeze!, unfreeze!, validate, gradients, stop


# Callback niceties from Flux.jl
call_fn(f, xs...) = f(xs...)
runall(f) = f
runall(fs::AbstractVector) = () -> foreach(call_fn, fs)


"""
The Workout keeps track of the progress of the training session. At least a model
and a loss function needs to be provided. Optional an optimizer and one or more
metrics can be specified.

If no optimizer is specified, SGD will be used. If no metrics are provided, only
the loss during training and validation will be registered (:loss and :val_loss).

The provided mover will move data to the correct device. See also SmartMover. If
no mover is required, you can provide: (x) -> x or simple identity

# Usage

```julia
workout = Workout(model, L1Loss())

workout = Workout(model, CrossEntropy(), Adam())

workout = Workout(model, HingeLoss(); acc=BinaryAccuracy())

workout = Workout(model, L1Loss(), mover=identity)
```
"""
mutable struct Workout
    model::Layer
    loss::Union{Loss,Function}
    opt
    metrics::Vector{Pair}
    history::IdDict{Symbol,MetricStore}
    steps::Int
    epochs::Int
    mover::Union{Mover,Function}

    function Workout(model::Layer, loss::Union{Loss,Function},
                     opt=Knet.SGD(); mover=SmartMover(), metrics...)
        new(model, loss, opt, collect(metrics), IdDict(), 0, 0, mover)
    end
end


"""
Save a workout to a file. This will save all the state that is captured in the workout
and enables to continue at a later stage using the loadWorkout function. Under the hood
this function uses Julia serialization.
"""
function saveworkout(workout::Workout, filename="workout_$(workout.steps).dat")::String
    Serialization.serialize(filename, workout)
    return filename
end

"""
Load a workout from a file and return the initialized Workout.

# Usage

```julia
workout = loadWorkout("workout_1000.dat")
train!(workout, mydata)
```
"""
function loadworkout(filename)::Workout
    workout = Serialization.deserialize(filename)
    return workout
end


struct StopException <: Exception end

"""
Stop a training session. Typically invoked by a callback function that detects
that the training is not progressing anymore.

If this function is called outside the scope of a trianing session, an exception is thrown.
"""
stop(workout::Workout, reason::String) = throw(StopException())


"""
Checks if the workout has any recorded values for a certain metric. The provided
metricname has to be fully qualified.

# Usage

```julia
if hasmetric(workout, :val_loss) ...
```

"""
hasmetric(workout::Workout, metricname::Symbol)::Bool = haskey(workout.history, metricname)


"""
Update the workout history with a single metric value.
"""
function updatemetric!(workout::Workout, metricname::Symbol, value)
    e = get!(workout.history, metricname, SmartReducer())
    update!(e, workout.steps, value)
end

"""
Invoke the configured metrics. The loss metric will always be logged and available.
Metrics are stored in the history attribute of the workout.
"""
function updatemetrics!(workout::Workout, loss::Number, y, y_pred, phase=:train)

    # First register the loss
    metricname = getmetricname(:loss, phase)
    e = get!(workout.history, metricname, SmartReducer())
    update!(e, workout.steps, loss)

    # And now run and register any additional metrics
    for (name,fn) in workout.metrics
        try
            metricname = getmetricname(name, phase)
            metricvalue = fn(y_pred, y)
            updatemetric!(workout, metricname, metricvalue)
        catch
            @warn "Failed executing metric." metricname maxlog=1
        end
    end
    return loss
end

"""
Get the metric value for a fully qualified metric name and a certain step. If
step is not provided the last step will be used. If no value is found the passed
function will not be invoked.

# Usage

```julia
getmetricvalue(workout, :val_loss) do value
    println("validation loss", value)
end
```
"""
function getmetricvalue(f::Function, workout::Workout, metricname::Symbol, step=workout.steps)
    if haskey(workout.history, metricname)
        m = workout.history[metricname]
        value = get(m.state, step, nothing)
        value !== nothing && f(value)
    end
end


"""
Get the history of a metric. The provided metric has the fully qualified name
and the returned value is a tuple of steps and values.

```julia
h = history(workout, :val_loss)
# returns e.g ([1000, 2000, 3000, 4000], [0.81, 0.73, 0.64, 0.61])

h = history(workout, :loss)
```

"""
function history(workout::Workout, metric::Symbol)::Tuple
      h = workout.history[metric].state
      steps = sort(collect(keys(h)))
      return (steps, [h[step] for step in steps])
end


"""
Utility function to calculate the gradients. Useful when checking for vanishing or
exploding gradients. The returned value is a Vector of (Param, Gradient).
"""
function gradients(workout::Workout, minibatch)
    x, y = workout.mover(minibatch)
    J = Knet.@diff begin
        y_pred = workout.model(x)
        workout.loss(y_pred, y)
    end
    gradients = []
    for p in Knet.params(J)
        p.opt === :NOUPDATE && continue
        if p.opt === nothing; p.opt = Knet.clone(workout.opt); end
        push!(gradients, (p, Knet.grad(J,p)))
    end
    return gradients
end

"""
Freeze a parameter so it no longer will be updated during training.
"""
freeze!(p::Knet.Param) = p.opt = :NOUPDATE

"""
Unfreeze a parameter so it will be updated again during training.
"""
unfreeze!(p::Knet.Param)= p.opt = nothing


"""
Perform the back propagation and update of weights in a single go.
"""
function back!(J, opt)
    for p in Knet.params(J)
        p.opt === :NOUPDATE && continue
        if p.opt === nothing; p.opt = Knet.clone(opt); end
        Knet.update!(p, Knet.grad(J,p))
    end
end


"""
Take a single step in updating the weights of a model. This function
will be invoked from train! to do the actual learning.

For a minibatch (x,y) of data, the folowing sequence will be executed:

1. perform the forward pass
2. calculate the loss
3. update and remember the metrics, if any
4. do the backpropagation and update the weights
"""
function step!(workout::Workout, minibatch; zerograd=true)
    workout.steps += 1
    x, y = workout.mover(minibatch)
    J = Knet.@diff begin
        y_pred = workout.model(x)
        loss = workout.loss(y_pred, y)
        updatemetrics!(workout, Knet.value(loss), y, y_pred)
        loss
    end
    back!(J, workout.opt)
end


"""
Predict a sample, either a single value or a batch. Compared to invoking
the model directory with model(x), predit takes care of:

- Moving the data to the GPU if required.
- Shaping the data into a batch (controlled by makebatch parameter)

# Usage

```julia
x = randn(Float32, 224, 224, 3)
predict(workout, x)
```
"""
function predict(workout, x; makebatch=true)
    x = makebatch ? addlast(x) : x
    y = workout.model(workout.mover(x))
    makebatch ? droplast(y) : y
end


"""
Validate a minibatch and calculate the loss and metrics. Typically this function
is called from the train! method. But if required can also be invoked directly.
"""
function validate(workout::Workout, minibatch)
    x, y = workout.mover(minibatch)
    y_pred = workout.model(x)
    loss = workout.loss(y_pred, y)
    updatemetrics!(workout, loss, y, y_pred, :valid)
end


"""
Train the model based on a supervised dataset and for a number
of epochs. train! can be called multiple times and will continue
to train where is left of last time.

By default the train! function will try to ensure the data is of the right
type (e.g. Float32) and on the right device (e.g. GPU) before feeding it to
the model.

# Usage

```julia
train!(workout, traindata)
train!(workout, traindata, testdata, epochs=50)
```

"""
function train!(workout::Workout, data, validation=nothing;
    epochs=1, cb = nothing)

    cb = isnothing(cb) ? Photon.Callbacks.ConsoleMeter() : cb
    cb = runall(cb)

    for epoch in 1:epochs
        workout.epochs += 1
        d = data isa Function ? data() : data
        for minibatch in d
            step!(workout, minibatch)
            cb(workout, :train)
        end

        if validation !== nothing
            d = validation isa Function ? validation() : validation
            for minibatch in d
                validate(workout, minibatch)
            end
        end

        updatemetric!(workout, :epoch, workout.epochs)

        try
            cb(workout, :valid)
        catch ex
            ex isa StopException ? break : rethrow(ex)
        end
    end

end

@debug "Loaded Training module"
