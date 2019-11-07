using Photon, Serialization
import Knet

function saveWorkout(workout::Workout, filename="workout.sav")

    ps = Knet.params(workout.model)

    # Lets not store gradients
    Photon.zerograd!(ps)

    # Move weights to CPU so the serialization contains all
    tmp_ps = IdDict()
    for p in ps
        tmp_ps[p] = p.value
        p.value = Array(p.value)
    end

    # serialize
    serialize(filename, workout)

    # set back the parameters to their orginal device
    for p in ps
        p.value = tmp_ps[p]
    end

    return filename
end

function loadWorkout(filename="workout.sav"; convertor=Photon.autoConvertor)::Workout

    # serialize
    workout = deserialize(filename)

    # set back the parameters to their orginal device
    ps = Knet.params(workout.model)
    for p in ps
        p.value = convertor(p.value)
    end
    return workout
end



m = Sequential(
        Dense(20),
        Dense(10)
        )

m(randn(10,16))

workout = Workout(m, MSELoss(), SGD())

saveWorkout(workout)

workout2 = loadWorkout()

workout

workout2
