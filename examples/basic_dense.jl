using Photon
using Photon.Layers
using Photon.Losses: CrossEntropyLoss

# Define a model with fully connected layers.
model = Sequential(
      Dense(256, :relu),
      Dense( 10),
      Softmax(),
)

# create the workout
workout = Workout(model, CrossEntropyLoss(); mover= SmartMover(Array{Float64}))

# create some dummy training data
X = [randn(25, 16) for i in 1:10]
Y = [rand(10, 16) for i in 1:10]

# perform the training
train!(workout, zip(X,Y))


# We can call the trained model directly to predict a minibatch of 16 samples
x = randn(25, 16)
y = model(x)

# Or use the predict function for a single sample.
predict(workout, randn(25))


println("### Done ###")
