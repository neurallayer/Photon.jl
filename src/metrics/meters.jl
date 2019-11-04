
export TensorBoardMeter, ConsoleMeter

"""
A meter is reponsible for presenting metric values. This can be
printing it to the console output, showing it on TensorBoard of storing it
in a database.
"""

mutable struct TensorBoardMeter <: Meter
    logger
    path::String
    metrics
    last_processed::IdDict{Symbol,Int}

    function TensorBoardMeter(path="./tensorboard_logs/runs", metrics=[:loss, :valid_loss])
        try
            @eval import TensorBoardLogger
        catch
            @warn "Package TensorBoardLogger not installed"
        end
        new(nothing, path, metrics, IdDict())
    end
end

function display(meter::TensorBoardMeter, workout::Workout, phase)
    if meter.logger == nothing
        meter.logger = TensorBoardLogger.TBLogger(meter.path)
    end

    for metric in meter.metrics
        getmetricvalue(workout, metric) do value
            last = get(meter.last_processed, metric, -1)
            workout.steps > last && TensorBoardLogger.log_value(meter.logger,
                string(metric), value, step=workout.steps)
            meter.last_processed[metric] = workout.steps
        end
    end
end


mutable struct ConsoleMeter <: Meter
    throttle::Float64
    next::Float64
    metricnames::Vector{Symbol}

    ConsoleMeter(throttle=1.0) = new(throttle, 0.0, [:loss, :valid_loss])
end


function display(meter::ConsoleMeter, workout::Workout, phase=:train)
    now = time()
    if now > meter.next || phase == :valid
        result = ""
        for metricname in meter.metricnames
            getmetricvalue(workout, metricname) do value
                    result *= @sprintf  " %s=%2.6f" metricname value
            end
        end
        if result != ""
            s = @sprintf "epoch:%4d step:%7d |" workout.epochs workout.steps
            print("\r", s, result)
            meter.next = now + meter.throttle
        end
        phase == :valid && println()
    end
end
