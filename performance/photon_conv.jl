using Photon.Layers
using Photon.Losses: MSELoss
using Photon

function test_performance()

  model = Sequential(
    Conv2D(16, (3,3), :relu),
    Conv2D(16, (3,3), :relu),
    BatchNorm(),
    MaxPool2D(4),
    Conv2D(64, (3,3), :relu),
    Conv2D(64, (3,3), :relu),
    BatchNorm(),
    MaxPool2D(4),
    Conv2D(256, (3,3), :relu),
    Conv2D(256, (3,3), :relu),
    BatchNorm(),
    MaxPool2D(4),
    Dense(64, :relu),
    Dense(10, :relu)
  )

  # Create dummy data
  X = (randn(224,224,3,16) for _=1:100)
  Y = (randn(10,16) for _=1:100)
  data = collect(zip(X,Y))

  # Create the workout
  workout = Workout(model, MSELoss())

  # One run ensure all is compiled
  train!(workout, data)

  # 10 runs we going to measure
  @time train!(workout, data, epochs=10)
end

test_performance()


