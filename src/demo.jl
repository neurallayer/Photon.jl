using Photon
using Knet

# A simple convolutional network
model = Sequential(
    Conv2D(32,3,activation=relu),
    MaxPool2D(),
    Flatten(),
    Dense(50,activation=relu),
    Dense(10)
)


# Lets create 100 batches of random images and labels
x = [randn(28,28,1,16) for _ in 1:100]
y = [rand(0:9,16) for _ in 1:100]

# Define the losss model
loss(x,y) = nll(model(x), y)

# And ready to train
adam!(loss, zip(x,y))
