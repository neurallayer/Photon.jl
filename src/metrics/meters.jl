
export TensorBoardMeter, ConsoleMeter

"""
A meter is reponsible for presenting metric values. This can be
printing it to the console output, showing it on TensorBoard of storing it
in a database.
"""

struct TensorBoardMeter <: Meter
    logger

    TensorBoardMeter(path="/tmp/runs") = new(TBLogger(path))
end


function display(meter::TensorBoardMeter, workout::Workout, prefix="")
    metricname = Symbol(prefix, :loss)
    value = workout.history[metricname]
    log_value(meter.logger,string(metric),value,step=workout.step)
end


mutable struct ConsoleMeter <: Meter
    throttle::Float64
    next::Float64

    ConsoleMeter(throttle=1.0) = new(throttle, 0.0)
end


function display(meter::ConsoleMeter, workout::Workout, prefix="")
    now = time()
    if now > meter.next
        metricname = Symbol(prefix, :loss)
        if haskey(workout.history, metricname)
            m = workout.history[metricname]
            value = get(m.state, workout.steps, nothing)
            if value != nothing
                s = @sprintf "epoch:%4d step:%7d | %s => %2.6f" workout.epochs workout.steps metricname value
                print("\r", s)
                meter.next = now + meter.throttle
            end
        end
    end
end
