using Photon, Knet

# Define a model with fully connected layers.
model = Sequential(
      Dense(256, activation = relu),
      Dense( 10),
      softmax,
)

# Lets make sure we use the cpu for this
ctx.devType=:cpu

# How to use the model to predict a minibatch
x = randn(Float32, 10, 16)
y = model(x)
