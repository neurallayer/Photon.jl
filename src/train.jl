import Base:haslength
import Serialization

export Workout, saveWorkout, loadWorkout, predict, fit!, hasmetric,
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

# Usage

```julia
workout = Workout(model, L1Loss())

workout = Workout(model, CrossEntropy(), Adam())

workout = Workout(model, HingeLoss(); acc=BinaryAccuracy())
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
    mover::Mover

    function Workout(model::Layer, loss::Union{Loss,Function},
                     opt=Knet.SGD(); mover=SmartMover(), metrics...)
        new(model, loss, opt, collect(metrics), IdDict(), 0, 0, mover)
    end
end


"""
Enable saving and loading of models by specialized KnetArray methods for Julia serialization
This will effectively move a GPU weight to the CPU before serialing it and move it back to
the GPU when deserializing.
"""
function Serialization.serialize(s::Serialization.AbstractSerializer, p::Knet.KnetArray)
    Serialization.serialize_type(s, typeof(p))
    Serialization.serialize(s, Array(p))
end

function Serialization.deserialize(s::Serialization.AbstractSerializer, t::Type{<:Knet.KnetArray})
    arr = Serialization.deserialize(s)
    return Knet.KnetArray(arr)
end


"""
Save a workout to a file. This will save all the state that is captured in the workout
and enables to continue at a later stage.
"""
function saveWorkout(workout::Workout, filename="workout_$(workout.steps).dat")::String
    # serialize
    Serialization.serialize(filename, workout)
    return filename
end

"""
Load a workout from file and return it.

# Usage

```julia
workout = loadWorkout("workout_1000.dat")
fit!(workout, mydata)
```
"""
function loadWorkout(filename)::Workout
    workout = Serialization.deserialize(filename)
    return workout
end


struct StopException <: Exception end

"""
Stop a training session. Typically invoked by a callback function that detects
that the training is not progressing anymore.

If this function is called outside the scope of a trianing session, an exception is thrown.
"""
function stop(workout::Workout, reason::String)
    throw(StopException())
end


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
Utility function to calculate the gradients. Useful when checking for vanishing or
exploding gradients. The returned value is a Vector of (Param, Gradient).
"""
function gradients(workout::Workout, minibatch=first(workout.dl))
    x, y = workout.mover(minibatch)
    J = Knet.@diff begin
        y_pred = workout.model(x)
        workout.loss(y_pred, y)
    end
    gradients = []
    for p in Knet.params(J)
        if p.opt === nothing; p.opt = Knet.clone(opt); end
        push!(gradients, (p, Knet.grad(J,p)))
    end
    return gradients
end

"""
Freeze a parameter so it no longer will be updated during training.
"""
function freeze!(p::Knet.Param)
    p.opt = :NOUPDATE
end

"""
Unfreeze a parameter so it will be updated again during training.
"""
function unfreeze!(p::Knet.Param)
    p.opt = nothing
end


"""
Perform the back propagation and update of weights in a single go.
"""
function back!(J::Knet.Tape, opt)
    for p in Knet.params(J)
        if p.opt === nothing; p.opt = Knet.clone(opt); end
        p.opt === :NOUPDATE && continue
        Knet.update!(p, Knet.grad(J,p))
    end
end


"""
Take a single step in updating the weights of a model. This function
will be invoked from fit! to do the actual learning.

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
- Making the data into a batch (controlled by makebatch parameter)

# Usage

```julia
x = randn(Float32, 224, 224, 3)
predict(model, x)
```
"""
function predict(workout, x; makebatch=true)
    x = makebatch ? addlast(x) : x
    y = workout.model(workout.conv(x))
    makebatch ? droplast(y) : y
end


"""
Validate a minibatch and calculate the loss and metrics. Typically this function
is called from the fit! method. But if required can also be invoked directly.
"""
function validate(workout::Workout, minibatch)
    x, y = workout.mover(minibatch)
    y_pred = workout.model(x)
    loss = workout.loss(y_pred, y)
    updatemetrics!(workout, loss, y, y_pred, :valid)
end


"""
Train the model based on a supervised dataset and for a number
of epochs. fit! can be called multiple times and will continue
to train where is left of last time.

By default the fit! function will try to ensure the data is of the right
type (e.g. Float32) and on the right device (e.g. GPU) before feeding it to
the model.

# Usage

```julia
fit!(workout, traindata)
fit!(workout, traindata, testdata, epochs=50)
```

"""
function fit!(workout::Workout, data, validation=nothing;
    epochs=1, cb = ConsoleMeter())
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
