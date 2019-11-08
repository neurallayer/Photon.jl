import Base:haslength
import Serialization

export Workout, saveWorkout, loadWorkout, predict, fit!, validate, hasmetric

"""
The Workout keeps track of the progress of a training session. At least a model
and a loss function needs to be provided. Optional an optimizer and one or more
metrics can be provided.

If no optimizer is provided, SGD will be used. If no metrics are provided only
the loss during training and validation will be registered (:loss and :val_loss).

Examples
========

    workout = Workout(model, mse)

    workout = Workout(model, nll, SGD())

    workout = Workout(model, nll, SGD(); acc=BinaryAccuracy())

"""
mutable struct Workout
    model::Layer
    loss::Function
    opt
    metrics::Vector{Pair}
    history::IdDict{Symbol,MetricStore}
    steps::Int
    epochs::Int

    function Workout(model::Layer, loss::Function, opt=Knet.SGD(); metrics...)
        new(model, loss, opt, collect(metrics), IdDict(), 0, 0)
    end
end


"""
Enable savign and loading of models by specialized functions for Julia serialization
"""
function Serialization.serialize(s::Serialization.AbstractSerializer, p::Knet.KnetArray)
    Serialization.serialize_type(s, typeof(p))
    Serialization.serialize(s, Array(p))
end

function Serialization.deserialize(s::Serialization.AbstractSerializer, t::Type{Knet.KnetArray})
    arr = Serialization.deserialize(s)
    return Knet.KnetArray(arr)
end


"""
Save a workout to file. This will save all the state that is captured in teh workout
and enables to continue at a later stage.
"""
function saveWorkout(workout::Workout, filename="workout.sav")
    # serialize
    Serialization.serialize(filename, workout)
    return filename
end

"""
Load a workout.
"""
function loadWorkout(filename="workout.sav")::Workout
    workout = Serialization.deserialize(filename)
    return workout
end

"""
Do the back propagation and update of weights in one go.
"""
function back!(J::Knet.Tape, opt)
    for p in Knet.params(J)
        if p.opt == nothing; p.opt = Knet.clone(opt); end
        Knet.update!(p, Knet.grad(J,p))
    end
end


"""
Does the workout have recorded values for a certain metric
"""
hasmetric(workout::Workout, metricname::Symbol) = haskey(workout.history, metricname)


"""
Function to determine the metric name.
"""
function getmetricname(metric::Symbol, phase=:train)::Symbol
    metricname = phase == :train ? metric : Symbol("val_", metric)
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
            e = get!(workout.history, metricname, SmartReducer())
            update!(e, workout.steps, metricvalue)
        catch
            @warn "Failed executing metric." metricname maxlog=1
        end
    end
    return loss
end

"""
Get the metric value for a fully qualified metric name and a certain step. If
step is not provided the last known step is used.

Examples:
=========

    getmetricvalue(workout, :val_loss) do value
        println("validation loss", value)
    end

"""
function getmetricvalue(f::Function, workout::Workout, metricname::Symbol, step=workout.steps)
    if haskey(workout.history, metricname)
        m = workout.history[metricname]
        value = get(m.state, step, nothing)
        value != nothing && f(value)
    end
end


function display(workout::Workout, meters, phase)
    for meter in meters
        try
            display(meter, workout, phase)
        catch
            @warn "Failed executing meter" meter maxlog=1
        end
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
function step!(workout::Workout, x, y; zerograd=true)
    workout.steps += 1
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

Examples:
=========

    x = randn(Float32, 224, 224, 3)
    predict(model, x)

"""
function predict(model, x; makebatch=true, convertor=autoConvertor)
    x = makebatch ? addlast(x) : x
    y = model(convertor(x))
    makebatch ? droplast(y) : y
end


"""
Validate a minibatch and calculate the loss and metrics. Typically this function
is called from the fit! method.
"""
function validate(workout::Workout, x, y)
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

Examples:
=========

    fit!(workout, traindata)
    fit!(workout, traindata, testdata, epochs=50)

If you don't want any data conversion, just pass the identity funciton
as the convertor:

    fit!(workout, traindata, convertor=identity)

"""
function fit!(workout::Workout, data, validation=nothing;
    epochs=1, convertor=autoConvertor, meters=[ConsoleMeter()])

    for epoch in 1:epochs
        workout.epochs += 1
        d = data isa Function ? data() : data
        for minibatch in d
            step!(workout, convertor(minibatch)...)
            display(workout, meters, :train)
        end

        if validation != nothing
            d = validation isa Function ? validation() : validation
            for minibatch in d
                validate(workout, convertor(minibatch)...)
            end
        end
        display(workout, meters, :valid)
    end
end

@info "Loaded Training module"
