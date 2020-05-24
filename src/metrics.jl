

"""
Stores the calculated metrics. If multiple values are provided at the same step
(like is the case with validation metrics), the moving average over those values
will be stored.
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
Function to generate the fully qualified metric name. It uses the metric name
and the phase (:train or :valid) to come up with a unique name.

```julia
getmetricname(:loss, :train) # return is :loss
getmetricname(:loss, :valid) # return is :val_loss
```
"""
function getmetricname(metric::Symbol, phase=:train)::Symbol
    metricname = phase == :train ? metric : Symbol("val_", metric)
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
Binary accuracy calculation
"""
struct BinaryAccuracy
    threshold

    BinaryAccuracy(;threshold=0.5) = new(threshold)
end

function (a::BinaryAccuracy)(y_pred::Tensor, y_true::Tensor)
    y_pred = y_pred .> a.threshold
    return mean(y_true .== y_pred)
end

"""
Binary accuracy for a onehot classification.
"""
struct OneHotBinaryAccuracy end

function (a::OneHotBinaryAccuracy)(y_pred::Tensor, y_true::Tensor)
    y_pred, y_true = Knet.mat(y_pred), Knet.mat(y_true)

    y_pred = mapslices(argmax,Knet.value(Array(y_pred)),dims=1)
    y_true = Array(y_true)

    mean(y_pred .== y_true)
end
