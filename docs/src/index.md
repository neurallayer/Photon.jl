# Photon

**Photon** is a developer friendly framework for Deep Learning in Julia.
Under the hood it leverages **Knet** and it provides a Keras like API on top of that.


## Steps
To train your own model, there are three steps to follow:

1) Create your model using the layers that come out of the box with Photon or using your own custom layers.

2) Create a *Workout* that combines the model, a loss function and an optimiser. Optionally you can also add some metrics that you want to monitor.

3) Train the model by calling fit! on the workout and the training data.


### Step 1: Model
A model can use out of the box layers or your own layers. Photon support most
common type of layer:

1) Dense or fully connected layers.
2) 2D ad 3D convolutional layers
3) Different type of Pooling layers
4) Recurrent layers (RNN, LSTM, GRU)
5) Dropout layers.
6) Several container type layers like Sequential and Residual


Some examples how to create the different type of models:

```julia
mymodel = Sequential(
            Dense(64, relu),
            Dense(10)
)
```

```julia
mymodel = Sequential(
            Conv2D(64, 3, relu),
            Conv2D(64, 3, relu),
            MaxPool(),
            Dense(10)
)
```

```julia
mymodel = Sequential(
            LSTM(64),
            Dense(64),
            Dense(10)
)
```


So normally you won't need to create your own layers. But if you have to, a layer
is nothing more than function. So the following could be a layer ;)

```julia
myLayer(X) = moon == :full ? X .- 1 : X
```


### Step 2: Workout
A workout combines a model + loss + optimiser and keeps track of the progress
during the actual training. The workout is stateful in the sense that you can run
multiple training sessions and the progress will be recorded appropriately.   

The minimum required to create a workout is:

```julia
workout = Workout(mymodel, MSE())
```

Besides this, you can pass an optimzer and define the metrics that you want to get tracked during
the training sessions. Photon tracks :loss and :val_loss (for the validation phase) by
default, but you define additional ones.

```julia
workout = Workout(mymodel, MSE(), SGD(), acc=accuracy())
```


### Step 3: fit!
The actual training is done using the fit! function.

```julia
fit!(workout, data, epochs=5)
```

If you provide also data for the validation phase, Photon will automatically run a validation after a training epoch has finished.


```julia
fit!(workout, data, training_data, epochs=10)
```

The data is expected to be a tuple of (X, Y) where X and Y can be tuples again in case
your model expects multiple inputs or outputs. So some examples of valid formats

```julia
(X,Y)
((X2,X2), Y)
(X, (Y1, Y2, Y3))
((X1, X2), (Y1, Y2))
```

By default fit! will convert each batch to the right data type and device. This is
controlled by the optional parameter *convertor*. If you don't want a convertor and
ensure the provided data is already in the right format, you can pass the identity function:

```julia
fit!(workout, data; convertor=identity)
```



## API

```@autodocs
Modules = [Photon]
Order   = [:function, :type]
```
