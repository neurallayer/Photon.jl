
module PhotonTest

using Photon
using Knet: relu


function test()

  model = Sequential(
    Conv2D(16, (3,3), activation=relu),
    Conv2D(16, (3,3), activation=relu),
    BatchNorm(),
    MaxPool2D(pool_size=4),
    Conv2D(64, (3,3), activation=relu),
    Conv2D(64, (3,3), activation=relu),
    BatchNorm(),
    MaxPool2D(pool_size=4),
    Conv2D(256, (3,3), activation=relu),
    Conv2D(256, (3,3), activation=relu),
    BatchNorm(),
    MaxPool2D(pool_size=4),
    Dense(64, activation=relu),
    Dense(10, activation=relu)
  )

  ctx.devType = :gpu
  data = randn(Float32,224,224,3,16)

  model(KorA(data))

  @time for i in 1:1000
    X = KorA(data)
    model(X)
  end
end

test()

end
