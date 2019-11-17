

"""
Save the Workout at the end of every epoch.

# Usage
```julia
fit!(workout, data, cb=EpochSave())
```

"""
struct EpochSave
end

function (c::EpochSave)(workout::Workout, phase::Symbol)
    if phase == :valid
        saveWorkout(workout)
    end
end



"""
Save the Workout if a certain metric has improved

# Usage
```julia

# save as long as the validation loss is declining
fit!(workout, data, cb=AutoSave(:val_loss))
```

"""
mutable struct AutoSave
    value::Float64
    metric::Symbol
    AutoSave(metric::Symbol) = new(Inf, metric)
end

function (c::AutoSave)(workout::Workout, phase::Symbol)
    if phase == :valid
        getmetricvalue(workout, c.metric) do x
            if x < c.value
                saveWorkout(workout)
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
