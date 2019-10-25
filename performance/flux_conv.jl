
module FluxTest

using Flux
using Flux:cu


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
    Dense(64, 10, relu),
  ) |> gpu


  data = randn(Float32,224,224,3,16)

  model(cu(data))

  @time for i in 1:1000
    X = cu(data)
    model(X)
  end
end

test()

end
