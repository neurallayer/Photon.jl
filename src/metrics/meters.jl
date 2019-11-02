

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
    throttle::Float
    last::Float

    TensorBoardMeter(throttle::Float=1) = new(throttle, 0.0)
end


function display(meter::ConsoleMeter, workout::Workout, phase="")
    metricname = Symbol(prefix, :loss)
    value = workout.history[metricname]
    log_value(meter.logger,string(metric),value,step=workout.step)
end
