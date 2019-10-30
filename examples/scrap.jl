using Photon
using Knet:relu, softmax
using Statistics

import Knet


# Create a simple COnvolutional network
model = Sequential(
      Conv2D(16, 3, relu),
      Conv2D(16, 3, relu),
      MaxPool2D(),
      Dense(256, relu),
      Dense(10),
      softmax
)


# Create a workout containing the model, a loss function and the optimizer
workout = Workout(model, bce_loss, ADAM())


X = [randn(Float32, 28,28,1,100) for _=1:600]
Y = [rand(0:1,10,100) for _=1:600]

# Run the training for 10 epochs
fit!(workout, zip(X,Y); epochs=10)

println("Trained the model in $(workout.epochs) epochs.")
