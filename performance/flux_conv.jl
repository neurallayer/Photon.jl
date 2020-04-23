
module FluxTest

using Flux
using CuArrays
using Flux: @epochs

function test()
  model = Chain(
    Conv((3,3), 3=>16, relu),
    Conv((3,3), 16=>16, relu),
    BatchNorm(3),
    MaxPool((4,4)),
    Conv((3,3), 16=>64, relu),
    Conv((3,3), 64=>64, relu),
    BatchNorm(3),
    MaxPool((4,4)),
    Conv((3,3), 64=>256, relu),
    Conv((3,3), 256=>256, relu),
    BatchNorm(3),
    MaxPool((4,4)),
    x -> reshape(x, :, size(x, 4)),
    Dense(1024, 64, relu),
    Dense(64, 10, relu)
  ) |> gpu


  # Create dummy data. This is a bit cheating since all the data is already
  # put on the GPU. Ideally this should happen only during training to simulate
  # large datasets.
  X = (cu(randn(Float32,224,224,3,16)) for _=1:100)
  Y = (cu(randn(Float32,10,16)) for _=1:100)
  data = collect(zip(X,Y))

  loss(x, y) = Flux.mse(model(x), y)
  opt = Descent()

  # One training epochs to get everything compiled
  Flux.train!(loss, params(model), data, opt)

  # Now it is time to measure
  @time @epochs 10 Flux.train!(loss, params(model), data, opt)

end

test()

end
