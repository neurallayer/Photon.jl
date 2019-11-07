using Photon

m = Sequential(
        Dense(20),
        Dense(10)
        )


X = randn(Float32,10,16)
pred = m(X)

workout = Workout(m, MSELoss(), SGD())

saveWorkout(workout)

workout2 = loadWorkout()

pred2 = workout2.model(X)

pred2 == pred
