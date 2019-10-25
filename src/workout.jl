'''
Keeps track of the status of the training sessions of a model.
Workout supports saving the in-between results to disk and continue
at a later time.
'''
mutable struct Workout
    model
    opt
    loss
    metrics
    steps
    epochs

    function Workout(model, opt, loss; metrics=[])
        new(model, opt, loss, metrics, 0, 0)
    end
end

'''
Take a single step in training the model and upgrading the weights.
'''
function step(workout::Workout, X, Y)
    y_pred = workout.model(X)
    loss = workout.loss(y_pred, Y)
    grad = grad(loss)
    update!(loss)
    workout.steps += 1
end


function fit!(workout, data; epochs=1)
    for epoch in 1:epochs
        workout.epochs += 1
        for minibatch in data
            loss = step(workout, minibatch...)
        end

    end
end
