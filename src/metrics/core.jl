
export SmartReducer, history, BinaryAccuracy

"""
Stores the calculated metrics
"""


struct SmartReducer <: MetricStore
    state::Dict{Int, Real}
    momentum::Real
    SmartReducer(momentum=0.9) = new(Dict(), momentum)
end

function update!(r::SmartReducer, step::Int, value::Real)
    if haskey(r.state, step)
        r.state[step] = r.momentum * r.state[step] + (1-r.momentum) * value
    else
        r.state[step] = value
    end
end



"""
Get the history of a metric.
"""
function history(workout::Workout, symbol::Symbol)
      h = workout.history[symbol].state
      steps = sort(collect(keys(h)))
      return (steps, [h[step] for step in steps])
end


"""
Binary accuracy calculation
"""
struct BinaryAccuracy
    name::Symbol
    threshold

    BinaryAccuracy(;threshold=0.5, name=:acc) = new(name,threshold)
end

function (a::BinaryAccuracy)(y_pred, y_true)
    y_pred = y_pred .> a.threshold
    return mean(y_true .== y_pred)
end
