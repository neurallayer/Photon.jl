
"""
struct TensorBoardMeter
    logger

    TensorBoardMeter(path="/tmp/runs") = new(TBLogger(path))
end


function display(meter::TensorBoardMeter, workout::Workout)
    metrics = [:loss, :valid_loss]
    value = workout[:]
    log_value(meter.logger,string(metric),value,step=workout.step)
end
"""
