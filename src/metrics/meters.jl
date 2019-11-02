

"""
A meter is reponsible for presenting metric values. This can be
printing it to the console output, showing it on TensorBoard of storing it
in a database.
"""
abstract type Meter end



struct TensorBoardMeter <: Meter
    logger

    TensorBoardMeter(path="/tmp/runs") = new(TBLogger(path))
end


function display(meter::TensorBoardMeter, workout::Workout, phase="")
    metricname = Symbol(prefix, :loss)
    value = workout.history[metricname]
    log_value(meter.logger,string(metric),value,step=workout.step)
end




struct ConsoleMeter <: Meter
    throttle::Float64
    next::Float64

    TensorBoardMeter(throttle=1.0) = new(throttle, 0.0)
end


function display(meter::ConsoleMeter, workout::Workout, phase="")
    now = time()
    if now > meter.next
        metricname = Symbol(prefix, :loss)
        a = get(workout.history[metricname],workout.step) do
            println(metricname, " => ", a)
        end
        meter.next = now + (throttle * 1000.0)
    end
end
