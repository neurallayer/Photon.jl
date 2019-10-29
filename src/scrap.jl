

using Photon
using Knet: relu


model = Sequential(
  Conv2D(16, 3, activation=relu),
  Conv2D(16, 3, activation=relu),
  BatchNorm(),
  MaxPool2D(pool_size=4),
  Conv2D(64, 3, activation=relu),
  Conv2D(64, 3, activation=relu),
  BatchNorm(),
  MaxPool2D(pool_size=4),
  Conv2D(256, 3, activation=relu),
  Conv2D(256, 3, activation=relu),
  BatchNorm(),
  MaxPool2D(pool_size=4),
  Dense(64, activation=relu),
  Dense(10, activation=relu)
)


X = [randn(Float32,224,224,3,16) for _ = 1:20]
Y = [randn(Float32,10,16) for _ = 1:20]

workout = Workout(model, mse, SGD())

fit!(workout, zip(X,Y), epochs=5)

predict(model, first(X), true)
