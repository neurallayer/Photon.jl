
export SmartReducer, history

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

"""
Get the history of a metric.

"""
function history(workout::Workout, symbol::Symbol)
      h = workout.metrics[symbol].history
      steps = sort(collect(keys(h)))
      return (steps, [h[step] for step in steps])
end
