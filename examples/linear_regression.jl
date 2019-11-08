using Photon, Printf
using Knet:value

"""
The most simple model possible, just a single layer with 1 output
"""
model = Dense(1)


"""
This is the function we want to approximate
"""
fn(x,y,z) = 2x - 3y + 4z .- 5


"""
Now we create the training data
"""
X = [randn(Float32, 3,8) for _ in 1:50]
Y = [fn(x[1,:], x[2,:], x[3,:]) for x in X]

"""
And finally we create the workout and start the training.
We use Mean Square Error loss and SGD as the optimizer.
"""
workout = Workout(model, mse)
fit!(workout, zip(X,Y), epochs=5)

@printf "%.1fx + %.1fy  + %.1fz + %.1f\n" Array(value(model.params.w))... Array(value(model.params.b))...
