
mutable struct Workout
    model
    loss
    opt
    metrics
    steps
    epochs

    function Workout(model, loss, opt; metrics=[])
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

"""
Take a single step in updating the model, so for the minibatch x,y
do:
- the forward pass
- calculate the loss
- do the backpropagation
- and update the weights of the model
"""
function step!(workout::Workout, x, y)
    J = @diff workout.loss(workout.model(x),y)
    ps = back(J)
    update!(workout.opt, ps)
    workout.steps += 1
end


"""
Train the model based on a supervised dataset and the number of
epochs to run.
"""
function fit!(workout, data, epochs::Int=1)

    for epoch in 1:epochs
        for (x,y) in data
            step!(workout, x, y)
        end
        workout.epochs += 1
    end
end

@info "Loaded Training module"
