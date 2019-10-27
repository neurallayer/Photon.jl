

struct SmartReducer
    history
    momentum
    SmartReducer(momentum=0.9) = new(Dict(), momentum)
end

function update!(r::SmartReducer, step::Int, value::Number)
    if haskey(r.history, step)
        r.history[step] = r.momentum * r.history[step] + (1-r.momentum) * value
    else
        r.history[step] = value
    end
end




mutable struct Workout
    model
    loss
    opt
    metrics
    steps
    epochs

    function Workout(model, loss, opt; metrics=Dict())
        new(model, loss, opt, metrics, 0, 0)
    end
end


"""
Perform the backpropagation and update the gradients. The weights are not yet
updated, that is the role of an optimiser.
"""
function back(J)
    ps = params(J)
    for param in ps
        # ugly hack, reuse opt attribute for storing gradients ;)
        param.opt = grad(J,param)
    end
    ps
end


function updatemetrics(workout, loss, y, y_pred, prefix="")
    metricname = Symbol(prefix, "loss")
    e = get!(workout.metrics, metricname, SmartReducer())
    update!(e, workout.steps, value(loss))
    return loss
end


"""
Take a single step in updating the weights of a model. So for the minibatch (x,y)
the folowing sequence will be executed:

1. the forward pass
2. calculate the loss
3. Calculate additional metrics, if any
4. do the backpropagation and calculate the gradients
5. update the weights of the model
"""
function step!(workout::Workout, x, y)
    workout.steps += 1
    J = @diff begin
        y_pred = workout.model(x)
        loss = workout.loss(y_pred, y)
        updatemetrics(workout, loss, y, y_pred)
    end
    ps = back(J)
    update!(workout.opt, ps)
end


"""
Predict a minibatch and calculate the loss and defined metrics
"""
function predict(workout::Workout, x, y)
    y_pred = workout.model(x)
    loss = workout.loss(y_pred, y)
    updatemetrics(workout, loss, y, y_pred, "valid_")
end


"""
Train the model based on a supervised dataset and the number of
epochs to run.
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
