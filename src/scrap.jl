
using Flux
using Flux:CuArray

model = Chain(
  Conv((3,3), 3=>16, relu),
  Conv((3,3), 16=>16, relu),
  MaxPool((4,4)),
  Conv((3,3), 16=>64, relu),
  Conv((3,3), 64=>64, relu),
  MaxPool((4,4)),
  Conv((3,3), 64=>256, relu),
  Conv((3,3), 256=>256, relu),
  MaxPool((4,4)),
  x -> reshape(x, :, size(x, 4)),
  Dense(1024, 64, relu),
  Dense(64, 10, relu),
) |> gpu


data = randn(Float32,224,224,3,16)

@time for i in 1:10000
  X = CuArray(data)
  model(X)
end

6.228136 seconds (1.61 M allocations: 77.377 MiB, 1.17% gc time)


using Photon
using Knet


model = Sequential(
  Conv2D(16, (3,3), activation=relu),
  Conv2D(16, (3,3), activation=relu),
  MaxPool2D((4,4)),
  Conv2D(64, (3,3), activation=relu),
  Conv2D(64, (3,3), activation=relu),
  MaxPool2D((4,4)),
  Conv2D(256, (3,3), activation=relu),
  Conv2D(256, (3,3), activation=relu),
  MaxPool2D((4,4)),
  Dense(64, activation=relu),
  Dense(10, activation=relu),
  )

ctx.devType = :gpu
data = randn(Float32,224,224,3,16)

model(KnetArray(X))

@time for i in 1:10000
  X = KnetArray(data)
  model(X)
end

4.043952 seconds (1.24 M allocations: 46.455 MiB, 0.82% gc time)

loss(x, y) = crossentropy(model(x), y)

X = [CuArray(randn(Float32,768,16)) for i in 1:100]
Y = [CuArray(rand(Float32,10,16)) for i in 1:100]
data = zip(X,Y)

 opt = ADAM()

@time for epoch in 1:10
  Flux.train!(loss, params(model), data, opt)
end



struct Dense
  W
end

(l::Dense)(X) = l.W .* X

mylayer = Dense(param(randn(Float32,10,10)))

mylayer(randn(Float32, 10,10))


using Flux
using Flux:gradient

m = ConvTranspose((3,3), 2=>1)
x = rand(10,10,2,1)

gradient(()->sum(m(x)), params(m))

logitbinarycrossentropy(logŷ, y) = (1 - y)*logŷ - logσ(logŷ)

no_bias(args...) = Float32(0.0)

l = Dense(10,20,initb=no_bias)
l.W

l(randn(10,16))
l.W


function multiply(a,b)
  return a.*b
end


x = CuArray(randn(10,10))
multiply(x,x)



batchindex2(xs, i) = (reverse(Base.tail(reverse(axes(xs))))..., i)


input = [randn(Float32,28,28,1) for _ in 1:16]
data = randn(Float32,28,28,1,16)

@time for i in 1:100000
  batchindex2(data,2)
end
