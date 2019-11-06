import Base:haslength

export Workout, predict, fit!, validate, hasmetric

"""
The Workout keeps track of the progress of a training session.

Examples
========

    workout = Workout(model, mse, SGD())

    workout = Workout(model, nll, ADAM())

"""
mutable struct Workout
    model::Function
    loss::Function
    opt::Optimizer
    metrics::Vector{Pair}
    history::IdDict{Symbol,MetricStore}
    steps::Int
    epochs::Int

    function Workout(model::Function, loss::Function, opt::Optimizer; metrics...)
        new(model, loss, opt, collect(metrics), IdDict(), 0, 0)
    end
end


"""
Perform the back propagation and update the gradients. The weights are not yet
updated, that is the role of the optimizers. For now the gradients are stored
with the weights.
"""
function back(J::Knet.Tape)
    ps = Knet.params(J)
    for param in ps
        # ugly hack, reuse opt attribute for storing gradients ;)
        param.opt = Knet.grad(J,param)
    end
    ps
end


"""
Does the workout have recorded values for a certain metric
"""
hasmetric(workout::Workout, metricname::Symbol) = haskey(workout.history, metricname)

function getmetricname(metric::Symbol, phase=:train)::Symbol
    metricname = phase == :train ? metric : Symbol("val_", metric)
end


"""
Invoke the configured metrics. The loss metric will always be logged and available.
"""
function updatemetrics!(workout::Workout, loss::Number, y, y_pred, phase=:train)

    # First register the loss
    metricname = getmetricname(:loss, phase)
    e = get!(workout.history, metricname, SmartReducer())
    update!(e, workout.steps, loss)

    # And now run and register any additional metrics
    for (name,fn) in workout.metrics
        metricname = getmetricname(name, phase)
        metricvalue = fn(y_pred, y)
        e = get!(workout.history, metricname, SmartReducer())
        update!(e, workout.steps, metricvalue)
    end
    return loss
end

function getmetricvalue(f::Function, workout::Workout, metricname, step=workout.steps)
    if haskey(workout.history, metricname)
        m = workout.history[metricname]
        value = get(m.state, step, nothing)
        value != nothing && f(value)
    end
end

function display(workout::Workout, meters, phase)
    for meter in meters
        display(meter, workout, phase)
    end
end



"""
Reset the gradients of the provided paramters.
"""
zerograd!(ps) = for param in ps; param.opt = nothing; end

"""
Take a single step in updating the weights of a model. This function
will be invoked from fit! to do the actual learning.

For a minibatch (x,y) of data, the folowing sequence will be executed:

1. the forward pass
2. calculate the loss
3. Calculate additional metrics, if any
4. do the backpropagation and calculate the gradients
5. update the weights of the model
"""
function step!(workout::Workout, x, y; zerograd=true)
    workout.steps += 1
    J = Knet.@diff begin
        y_pred = workout.model(x)
        loss = workout.loss(y_pred, y)
        updatemetrics!(workout, Knet.value(loss), y, y_pred)
        loss
    end
    ps = back(J)
    update!(workout.opt, ps)
    zerograd && zerograd!(ps)
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
Validate a minibatch and calculate the loss and defined metrics.
"""
function validate(workout::Workout, x, y)
    y_pred = workout.model(x)
    loss = workout.loss(y_pred, y)
    updatemetrics!(workout, loss, y, y_pred, :valid)
end


"""
Train the model based on a supervised dataset and the number of
epochs to run. fit! can be called multiple times and will continue
where is left of last time.

Also it will try to make the data suitable for the model.

Examples:
=========

    fit!(workout, traindata)
    fit!(workout, traindata, testdata, epochs=50)

If you don't want any data conversion, just pass the identity funciton
as the convertor parameter:

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
