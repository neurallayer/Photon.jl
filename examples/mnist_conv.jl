using Photon
import Knet

# Get the MNIST data set
include(Knet.dir("data", "mnist.jl"))
trndata, tstdata = mnistdata()

# Create a simple Convolutional network
model = Sequential(
      Conv2D(16, 3, relu),
      Conv2D(16, 3, relu),
      MaxPool2D(),
      Dense(64, relu),
      Dense(10)
)

# Create a workout containing the model, a loss function and
# the optimizer (default SGD) and accuracy meter
workout = Workout(model, nll, acc=OneHotBinaryAccuracy())

# Run the training for 10 epochs and we don't need a convertor since
# mnistdata function already does the work.
fit!(workout, trndata, tstdata; epochs=10, convertor=identity)

println("\nTrained the model in $(workout.epochs) epochs.")

# Now let's plot some results. If you haven't installed Plots yet, you'll
# need to run:  using Pkg; Pkg.add("Plots")

import Plots
plotmetrics(Plots, workout, [:loss, :val_loss, :val_acc])

#= Plot the training loss
h1 = history(workout, :loss)
plot(h1..., xlabel = "steps", ylabel="values", label="loss")

# Plot the validation loss
h2 = history(workout, :val_loss)
plot!(h2..., linewidth = 2, label="validation loss")

# Plot the validation accuracy
h3 = history(workout, :val_acc)
plot!(h3..., linewidth = 2, label="validation accuracy")
=#
