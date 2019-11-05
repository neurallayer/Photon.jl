using Photon
using Knet:relu, softmax

# Define a model with fully connected layers.
model = Sequential(
      Dense(256, relu),
      Dense( 10),
      softmax,
)

# Lets use the CPU for this, even if we have a GPU.
setContext(device=:cpu)

# We can call the model to predict a minibatch
x = randn(Float32, 10, 16)
y = model(x)


# Or use the predict function for a single sample.
predict(model, randn(10))

println("### Done ###")
