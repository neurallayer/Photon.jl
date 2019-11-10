

"""
Save the Workout if a certain metric improved
"""
struct AutoSave
end

function (meter::AutoSave)(workout::Workout, phase::Symbol)
end


"""
Stop the training if a certain metric didn't improve
"""
struct EarlyStop
end

function (meter::EarlyStop)(workout::Workout, phase::Symbol)
end
