using Photon
using Knet


# A simple convolutional network
model = Sequential(
    ContextSwitch(devType=:gpu),
    Conv2D(32, 3, activation = relu),
    MaxPool2D(),
    Flatten(),
    ContextSwitch(devType=:cpu),
    Dense(50, activation = relu),
    Dense(10),
)


# Lets create 100 batches of random images and labels
x = [randn(Float64, 28, 28, 1, 16) for _ = 1:100]
y = [rand(0:9, 16) for _ = 1:100]

# Define the loss function
loss(x, y) = nll(model(x), y)

# And we are ready to train the model
adam!(loss, zip(x, y))

@info "Demo Done"
