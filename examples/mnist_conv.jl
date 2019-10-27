using Photon, Knet

# Get the MNIST data set
include(Knet.dir("data", "mnist.jl"))
trndata, tstdata = mnistdata()

# Create a simple COnvolutional network
model = Sequential(
      Conv2D(16, 3, activation = relu),
      Conv2D(16, 3, activation = relu),
      MaxPool2D(),
      Dense(256, activation = relu),
      Dense(10),
)

# Create a workout containing the model, a loss function and the optimizer
workout = Workout(model, nll, ADAM())

# Run the training for 10 epochs
fit!(workout, trndata, tstdata; epochs=10)

println("Trained the model in $(workout.epochs) epochs.")


# Not lets plot some results. If you haven't installed plots, you need to run
#     using Pkg; Pkg.add("Plots")

using Plots

function history(workout::Workout, symbol::Symbol)
      h = workout.metrics[symbol].history
      steps = sort(collect(keys(h)))
      return (steps, [h[step] for step in steps])
end


h1 = history(workout, :loss)
h2 = history(workout, :valid_loss)


plot(h1...)
plot!(h2...)
