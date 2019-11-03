
export TensorBoardMeter, ConsoleMeter

using TensorBoardLogger

"""
A meter is reponsible for presenting metric values. This can be
printing it to the console output, showing it on TensorBoard of storing it
in a database.
"""

struct TensorBoardMeter <: Meter
    logger
    metrics
    last_processed::IdDict{Symbol,Int}

    TensorBoardMeter(path="./runs", metrics=[:loss, :valid_loss]) =
        new(TBLogger(path), metrics, IdDict())
end


function display(meter::TensorBoardMeter, workout::Workout, phase)
    for metric in meter.metrics
        getmetricvalue(workout, metric) do value
            last = get(meter.last_processed, metric, -1)
            workout.steps > last && log_value(meter.logger,string(metric),value,step=workout.steps)
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
