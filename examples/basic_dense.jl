using Photon
using Knet:relu

# Define a model with fully connected layers.
model = Sequential(
      Dense(256, relu),
      Dense( 10),
      softmax,
)

# Lets use the cpu for this
setContext(device=:cpu)

# How to use the model to predict a minibatch
x = randn(Float32, 10, 16)
y = model(x)

println("### Done ###")
