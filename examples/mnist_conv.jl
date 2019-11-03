using Photon
using Knet:relu, nll, softmax
using Statistics

import Knet

# Get the MNIST data set
include(Knet.dir("data", "mnist.jl"))
trndata, tstdata = mnistdata()

# Create a simple COnvolutional network
model = Sequential(
      Conv2D(16, 3, relu),
      Conv2D(16, 3, relu),
      MaxPool2D(),
      Dense(64, relu),
      Dense(10)
)


# Create a workout containing the model, a loss function and the optimizer
workout = Workout(model, nll, ADAM())

# Run the training for 10 epochs and we don't need a convertor since
# mnist data already does the work.
fit!(workout, trndata, tstdata; epochs=10,
      convertor=identity, meters=[ConsoleMeter()])

println("Trained the model in $(workout.epochs) epochs.")


# Now let's plot some results. If you haven't installed Plots yet, you'll
# need to run:  using Pkg; Pkg.add("Plots")
using Plots

h1 = history(workout, :loss)
h2 = history(workout, :valid_loss)

plot(h1..., xlabel = "steps", ylabel="loss", label="training")
plot!(h2..., linewidth = 2, label="validation")
