
using Statistics

mutable struct Momentum
    momentum::Float64
    Momentum(momentum=0.9) = new(momentum)
end

function (c::Momentum)(x_prev, x_new)
    x_prev == nothing ?
        x_new : c.momentum*x_prev + (1-c.momentum)*x_prev
end


struct SmartReducer
    history
    momentum
    SmartReducer(momentum=0.9) = new(Dict(), momentum)
end

function update!(r:SmartReducer, step, metric, value)
    key = (step, metric)
    if haskey(r.history)
        r.history[key] = r.momentum * r.history[key] + (1-r.momentum) * value
    else
        r.history[key] = value
    end
end


mutable struct BinaryAccuracy

    name
    threshold
    value
    reducer

    BinaryAccuracy(;name="binary_accuracy", threshold=0.5, reducer=Momentum()) =
        new(name,threshold, nothing, reducer)
end


reset_states!(o::BinaryAccuracy) = o.value = nothing
result(o::BinaryAccuracy) = o.value
function update!(o::BinaryAccuracy, y_true, y_pred, step, epoch)
     y_pred = y_pred .> 0.5
     v = (y_true .* y_pred) .+ ((1 .- y_true) .* (1 .- y_pred))
     v = mean(v)
     o.value = o.reducer(o.value, v)
end

m = BinaryAccuracy()

y_pred = randn(10,16)
y_true = convert(Array{Float64}, rand(0:1,10,16))


y_true = [1, 1, 0, 0]
y_pred = [0.98, 1, 0, 0.6]

update!(m, y_true, y_pred, 1, 1)


(y_pred .> 0).*y_true

result(m)




y_true = [1, 1, 0, 0]
y_pred = [1, 1, 0, 0]
