
"""
The Workout keeps track of the progress of a training session.

Examples
========

    workout = Workout(model, mse, SGD())

    workout = Workout(model, nll, ADAM())

"""
mutable struct Workout
    model
    loss
    opt
    metrics
    steps::Int
    epochs::Int

    function Workout(model, loss, opt; metrics=Dict())
        new(model, loss, opt, metrics, 0, 0)
    end
end


"""
Perform the back propagation and update the gradients. The weights are not yet
updated, that is the role of the optimizers. For now the gradients are stored
with the weights.
"""
function back(J)
    ps = Knet.params(J)
    for param in ps
        # ugly hack, reuse opt attribute for storing gradients ;)
        param.opt = Knet.grad(J,param)
    end
    ps
end

"""
Invoke the configured metrics. The loss metric will always be logged and available.
"""
function updatemetrics(workout::Workout, loss, y, y_pred, prefix="")
    metricname = Symbol(prefix, "loss")
    e = get!(workout.metrics, metricname, SmartReducer())
    update!(e, workout.steps, Knet.value(loss))
    return loss
end


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
        updatemetrics(workout, loss, y, y_pred)
    end
    ps = back(J)
    update!(workout.opt, ps)
    zerograd && zerograd!(ps)
end


"""
Predict a minibatch and calculate the loss and defined metrics.
"""
function predict(workout::Workout, x, y)
    y_pred = workout.model(x)
    loss = workout.loss(y_pred, y)
    updatemetrics(workout, loss, y, y_pred, "valid_")
end


"""
Train the model based on a supervised dataset and the number of
epochs to run.

Examples:
========

    fit!(workout, traindata)
    fit!(workout, traindata, testdata, epochs=50)

"""
function fit!(workout::Workout, data, validation=nothing; epochs=1)

    for epoch in 1:epochs
        workout.epochs += 1
        d = data isa Function ? data() : data
        for (x,y) in d
            step!(workout, x, y)
        end

        if validation != nothing
            d = validation isa Function ? validation() : validation
            for (x,y) in d
                predict(workout, x, y)
            end
        end
    end
end

@info "Loaded Training module"
