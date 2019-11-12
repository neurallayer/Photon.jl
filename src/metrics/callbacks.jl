

"""
Save the Workout if a certain metric has improved
"""
struct AutoSave
    value
    metric::Symbol
    AutoSave(metric::Symbol) = new(1000, metric)
end

function (meter::AutoSave)(workout::Workout, phase::Symbol)
    if phase == :valid
        saveWorkout(workout)
    end
end


"""
Stop the training if a certain metric didn't improve
"""
struct EarlyStop
    value
    metric::Symbol
    EarlyStop(metric::Symbol) = new(1000, metric)
end

function (meter::EarlyStop)(workout::Workout, phase::Symbol)
    if phase == :valid
        saveWorkout(workout)
    end
end
