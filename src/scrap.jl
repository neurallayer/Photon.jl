
using Flux
using Printf

model = Chain(
          Dense(1, 10, relu),
          Dense(10, 1, relu))

N = 100
x = randn(1,16,N)
y = x.^2

loss(x, y) = Flux.mse(model(x), y)

opt = ADAM()
epochs = 20

ps = Flux.params(model)
tracked = Tracker.Params(ps)
@progress for epoch = 1:epochs
  for i = 1:N
    gs = Flux.Tracker.gradient(() -> loss(x[:,:,i], y[:,:,i]), ps)
    Flux.Tracker.update!(opt, tracked, gs)
  end
  @printf "Epoch: %d  3^2 = %1.4f\n" epoch model([2]).data[1]
end
