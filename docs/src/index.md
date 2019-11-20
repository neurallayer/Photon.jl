## Introduction
Photon is a developer friendly framework for Machine Learning in Julia.
Under the hood it leverages **Knet** and it provides a Keras like API on top of that.


## Installation
Since Photon is improving rapidly, the package can be best installed with the Julia package manager directly from the Github repository.

From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add https://github.com/neurallayer/Photon.jl
```

In the future it will be possible to use:

```
pkg> add Photon
```



## Steps
To train your own model, there are four steps to follow:

1) Create your **model** using the layers that come out of the box with Photon or using your own custom layers.

2) Create a **workout** that combines the model, a loss function and an optimiser. Optionally you can also add some metrics that you want to monitor.

3) Prepare your **data** with Data pipelines.

4) **Train** the model by calling fit! on the workout and the training data.


### Step 1: Create a Model
A model can use out of the box layers or your own layers. Photon support most
common type of layer:

1) Dense or fully connected layers
2) 2D ad 3D convolutional layers
3) Different type of Pooling layers
4) Recurrent layers (RNN, LSTM, GRU)
5) Dropout and Batch Normalization layers
6) Several types of container layers like Sequential and Residual


Some examples how to create the different type of network architectures:

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
myLayer(X) = is_full_moon() ? X .- 1 : X
```


### Step 2: Define a Workout
A workout combines a model + loss + optimiser and keeps track of the progress
during the actual training. The workout is stateful in the sense that you can run
multiple training sessions and the progress will be recorded appropriately.   

The minimum required code to create a workout is:

```julia
workout = Workout(mymodel, MSELoss())
```

This will create a workout that will use the default Optimiser (SGD) and only
the *loss metric* being tracked.

Alternatively you can pass an optimizer and define the metrics that you want to get tracked during the training sessions. Photon tracks :loss and :val_loss (for the validation phase) by default, but you can define additional ones.

```julia
workout = Workout(mymodel, MSE(), SGD(), acc=BinaryAccuracy())
```

Each additional metric is added as an optional parameter, where the name will be
the metricname and the function the metric calculation.

Another useful feature is that a Workout can saved and restored at any moment during the training. And not only the model and its parameters will be saved. Also the state of the optimiser and any logged metrics will be able saved and restored to their previous state. This even makes it also possible to shared workout with colleagues (although they need the same packages installed installed as you).

```julia
filename = saveWorkout(workout)

workout2 = loadWorkout(filename)
```

### Step 3: Prepare the Data
Although Photon is perfectly happy to accept plain Vectors as data, this often won't be feasible due to the data not fitting into memory. In those cases you can use the data pipeline feature of Photon.

A typical pipeline would look something like this:

```julia
  data = SomeDataset(source) |> SomeTransformers() |> MiniBatch()
```

Add then the pipeline can be used directly in the training cycle:

```julia
  fit!(workout, data)
```

Photon comes out the box with several reusable components for creating these pipelines. They can be divided into two types; *Datasets* that are the start of a pipeline and retrieve the data from some source and *Transformers* that transform the output of a previous step in the pipeline.

**Source datasets**
- ImageDataset
- TestDataset
- ArrayDataset
- JLDDataset

**Transformers**
- Normalizer
- Cropper
- MiniBatch
- Noiser

A complete pipeline for image data could look something like this:

```julia
data = ImageDataset(files, labels, resize=(250,250))
data = data |> Crop(200,200) |> Normalize() |> MiniBatch(8)
```

### Step 4: Run the Training
The actual training in Photon is done invoking the fit! function.

```julia
fit!(workout, data, epochs=5)
```

The validation phase is optional. But if you provide data for the validation phase, Photon will automatically run the validation step after each training epoch.


```julia
fit!(workout, data, training_data, epochs=10)
```
Defined metrics and loss will then be available both for training and validation.

The data is expected to be a tuple of (X, Y) where X and Y can be tuples again in case your model expects multiple inputs or outputs. So some examples of valid formats

```julia
(X,Y)
((X1,X2), Y)
(X, (Y1, Y2, Y3))
((X1, X2), (Y1, Y2))
```

By default fit! will convert each batch to the right data type and device. This is
controlled by the optional parameter *convertor*. If you don't want a conversion to take place and ensured the provided data is already in the right format, you can pass the identity function:

```julia
fit!(workout, data; convertor=identity)
```
