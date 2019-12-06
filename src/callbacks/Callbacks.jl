

module Callbacks


using ..Photon: stop, Workout, Tensor, getmetricvalue, saveWorkout, getContext
using Printf
using Statistics
import Knet

include("meters.jl")
export Meter, ConsoleMeter, SilentMeter, TensorBoardMeter, FileMeter, PlotMeter

export AutoSave, EpochSave, EarlyStop
"""
Save the Workout at the end of every epoch. Optionally provide a filename.

# Usage
```julia
fit!(workout, data, cb=EpochSave())
```

"""
struct EpochSave
    filename
    EpochSave(filename=nothing) = new(filename)
end

function (c::EpochSave)(workout::Workout, phase::Symbol)
    if phase == :valid
        if c.filename !== nothing
            saveWorkout(workout, c.filename)
        else
            saveWorkout(workout)
        end
    end
end



"""
Save the Workout if a certain metric has improved since the last epoch

# Usage
```julia

# save as long as the validation loss is declining
fit!(workout, data, cb=AutoSave(:val_loss))
```

"""
mutable struct AutoSave
    value::Float64
    metric::Symbol
    filename
    AutoSave(metric::Symbol, filename=nothing) = new(Inf, metric, filename)
end

function (c::AutoSave)(workout::Workout, phase::Symbol)
    if phase == :valid
        getmetricvalue(workout, c.metric) do x
            if x < c.value
                if c.filename !== nothing
                    saveWorkout(workout, c.filename)
                else
                    saveWorkout(workout)
                end
                c.value = x
            end
        end
    end
end


"""
Stop the training if a certain metric didn't improve
"""
mutable struct EarlyStop
    value::Float64
    metric::Symbol
    EarlyStop(metric::Symbol) = new(Inf, metric)
end

function (c::EarlyStop)(workout::Workout, phase::Symbol)
    if phase == :valid
        getmetricvalue(workout, c.metric) do x
            if x > c.value
                stop(workout,"$(c.metric) didn't improve anymore")
            else
                c.value = x
            end
        end
    end
end

end
