
module PhotonTest

using Photon


function test()

  model = Sequential(
    Conv2D(16, (3,3), relu),
    Conv2D(16, (3,3), relu),
    BatchNorm(),
    MaxPool2D(4),
    Conv2D(64, (3,3), relu),
    Conv2D(64, (3,3), relu),
    BatchNorm(),
    MaxPool2D(4),
    Conv2D(256, (3,3), relu),
    Conv2D(256, (3,3), relu),
    BatchNorm(),
    MaxPool2D(4),
    Dense(64, relu),
    Dense(10, relu)
  )

  data = randn(Float32,224,224,3,16)

  model(KorA(data))

  @time for i in 1:1000
    X = KorA(data)
    model(X)
  end
end

test()

end
