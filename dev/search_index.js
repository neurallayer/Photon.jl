var documenterSearchIndex = {"docs":
[{"location":"community/#Community-1","page":"Community","title":"Community","text":"","category":"section"},{"location":"community/#","page":"Community","title":"Community","text":"All Photon users are welcome to ask questions on the Julia forum. Of course issues can be opened on Github where you also can get the source code.","category":"page"},{"location":"#Photon-1","page":"Home","title":"Photon","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Photon is a developer friendly framework for Deep Learning in Julia. Under the hood it leverages Knet and it provides a Keras like API on top of that.","category":"page"},{"location":"#Steps-1","page":"Home","title":"Steps","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"To train your own model, there are three steps to follow:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Create your model using the layers that come out of the box with Photon or using your own custom layers.\nCreate a Workout that combines the model, a loss function and an optimiser. Optionally you can also add some metrics that you want to monitor.\nTrain the model by calling fit! on the workout and the training data.","category":"page"},{"location":"#Step-1:-Model-1","page":"Home","title":"Step 1: Model","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"A model can use out of the box layers or your own layers. Photon support most common type of layer:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Dense or fully connected layers.\n2D ad 3D convolutional layers\nDifferent type of Pooling layers\nRecurrent layers (RNN, LSTM, GRU)\nDropout layers.\nSeveral container type layers like Sequential and Residual","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Some examples how to create the different type of models:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"mymodel = Sequential(\n            Dense(64, relu),\n            Dense(10)\n)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"mymodel = Sequential(\n            Conv2D(64, 3, relu),\n            Conv2D(64, 3, relu),\n            MaxPool(),\n            Dense(10)\n)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"mymodel = Sequential(\n            LSTM(64),\n            Dense(64),\n            Dense(10)\n)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"So normally you won't need to create your own layers. But if you have to, a layer is nothing more than function. So the following could be a layer ;)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"myLayer(X) = moon == :full ? X .- 1 : X","category":"page"},{"location":"#Step-2:-Workout-1","page":"Home","title":"Step 2: Workout","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"A workout combines a model + loss + optimiser and keeps track of the progress during the actual training. The workout is stateful in the sense that you can run multiple training sessions and the progress will be recorded appropriately.   ","category":"page"},{"location":"#","page":"Home","title":"Home","text":"The minimum required to create a workout is:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"workout = Workout(mymodel, MSE())","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Besides this, you can pass an optimzer and define the metrics that you want to get tracked during the training sessions. Photon tracks :loss and :val_loss (for the validation phase) by default, but you define additional ones.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"workout = Workout(mymodel, MSE(), SGD(), acc=accuracy())","category":"page"},{"location":"#Step-3:-fit!-1","page":"Home","title":"Step 3: fit!","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"The actual training is done using the fit! function.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"fit!(workout, data, epochs=5)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"If you provide also data for the validation phase, Photon will automatically run a validation after a training epoch has finished.","category":"page"},{"location":"#","page":"Home","title":"Home","text":"fit!(workout, data, training_data, epochs=10)","category":"page"},{"location":"#","page":"Home","title":"Home","text":"The data is expected to be a tuple of (X, Y) where X and Y can be tuples again in case your model expects multiple inputs or outputs. So some examples of valid formats","category":"page"},{"location":"#","page":"Home","title":"Home","text":"(X,Y)\n((X2,X2), Y)\n(X, (Y1, Y2, Y3))\n((X1, X2), (Y1, Y2))","category":"page"},{"location":"#","page":"Home","title":"Home","text":"By default fit! will convert each batch to the right data type and device. This is controlled by the optional parameter convertor. If you don't want a convertor and ensure the provided data is already in the right format, you can pass the identity function:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"fit!(workout, data; convertor=identity)","category":"page"},{"location":"#API-1","page":"Home","title":"API","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Modules = [Photon]\nOrder   = [:function, :type]","category":"page"},{"location":"#Photon.GRU","page":"Home","title":"Photon.GRU","text":"Create a GRU layer\n\nExamples:\n\nlayer = GRU(50)\n\n\n\n\n\n","category":"function"},{"location":"#Photon.KorA-Tuple{Array}","page":"Home","title":"Photon.KorA","text":"KorA makes it easy to move an array to the GPU or the other way around\n\n\n\n\n\n","category":"method"},{"location":"#Photon.LSTM","page":"Home","title":"Photon.LSTM","text":"Create a LSTM layer.\n\nExamples:\n\nlayer = LSTM(50)\n\n\n\n\n\n","category":"function"},{"location":"#Photon.RNN","page":"Home","title":"Photon.RNN","text":"A simple RNN layer.\n\n\n\n\n\n","category":"function"},{"location":"#Photon.fit!","page":"Home","title":"Photon.fit!","text":"Train the model based on a supervised dataset and for a number of epochs. fit! can be called multiple times and will continue to train where is left of last time.\n\nBy default the fit! function will try to ensure the data is of the right type (e.g. Float32) and on the right device (e.g. GPU) before feeding it to the model.\n\nUsage\n\nfit!(workout, traindata)\nfit!(workout, traindata, testdata, epochs=50)\n\nIf you don't want any data conversion, just pass the identity funciton as the convertor:\n\nfit!(workout, traindata, convertor=identity)\n\n\n\n\n\n","category":"function"},{"location":"#Photon.hasmetric-Tuple{Workout,Symbol}","page":"Home","title":"Photon.hasmetric","text":"Does the workout have any recorded values for a certain metric\n\n\n\n\n\n","category":"method"},{"location":"#Photon.loadWorkout-Tuple{Any}","page":"Home","title":"Photon.loadWorkout","text":"Load a workout from file and return it.\n\nUsage\n\nworkout = loadWorkout(\"workout_1000.dat\")\nfit!(workout, mydata)\n\n\n\n\n\n","category":"method"},{"location":"#Photon.output_size-Tuple{Photon.Conv,Any}","page":"Home","title":"Photon.output_size","text":"Utility function to determine the output size of a convolutional layer given a certain input size and configuration of a convolutional layer. This function works even if the weights of the layer are not initialized.\n\nFor example: \tc = Conv2D(16, (3,3), strides=3, padding=1) \toutput_size(c, (224,224)) # output is (75, 75)\n\n\n\n\n\n","category":"method"},{"location":"#Photon.plotmetrics","page":"Home","title":"Photon.plotmetrics","text":"Plot the metrics after some training. This function will plot all the metrics in a single graph.\n\nIn order to avoid Photon being dependend on Plots, the calling code will have to provide that module as the first parameter.\n\nUsage\n\nfit!(workout, mydata, epochs=10)\n\nimport Plots\nplotmetrics(Plots, workout)\n\n\n\n\n\n","category":"function"},{"location":"#Photon.predict-Tuple{Any,Any}","page":"Home","title":"Photon.predict","text":"Predict a sample, either a single value or a batch. Compared to invoking the model directory with model(x), predit takes care of:\n\nMoving the data to the GPU if required.\nMaking the data into a batch (controlled by makebatch parameter)\n\nUsage\n\nx = randn(Float32, 224, 224, 3)\npredict(model, x)\n\n\n\n\n\n","category":"method"},{"location":"#Photon.saveWorkout","page":"Home","title":"Photon.saveWorkout","text":"Save a workout to a file. This will save all the state that is captured in thr workout and enables to continue at a later stage.\n\n\n\n\n\n","category":"function"},{"location":"#Photon.AdaptiveAvgPool","page":"Home","title":"Photon.AdaptiveAvgPool","text":"Adaptive Average Pool has a fixed size output and enables creating a convolutional network that can be used for multiple image formats.\n\n\n\n\n\n","category":"type"},{"location":"#Photon.AdaptiveMaxPool","page":"Home","title":"Photon.AdaptiveMaxPool","text":"Adaptive MaxPool has a fixed size output and enables creating a convolutional network that can be used for multiple image formats.\n\n\n\n\n\n","category":"type"},{"location":"#Photon.BCELoss","page":"Home","title":"Photon.BCELoss","text":"Binary CrossEntropy\n\n\n\n\n\n","category":"type"},{"location":"#Photon.BatchNorm","page":"Home","title":"Photon.BatchNorm","text":"BatchNorm layer with support for an optional activation function\n\n\n\n\n\n","category":"type"},{"location":"#Photon.Concurrent","page":"Home","title":"Photon.Concurrent","text":"Concurrrent\n\n\n\n\n\n","category":"type"},{"location":"#Photon.ContextSwitch","page":"Home","title":"Photon.ContextSwitch","text":"Beginning of allowing for a single model instance to run on multiple devices (expiremental)\n\n\n\n\n\n","category":"type"},{"location":"#Photon.CrossEntropyLoss","page":"Home","title":"Photon.CrossEntropyLoss","text":"CrossEntropy loss function with support for an optional weight parameter. The weight parameter can be static (for example to handle class inbalances) or dynamic (so passed every time when the lost function is invoked)\n\nUsage\n\nworkout = Workout(model, CE(), SGD())\n\n\n\n\n\n","category":"type"},{"location":"#Photon.Dense","page":"Home","title":"Photon.Dense","text":"Fully connected layer with an optional bias weight.\n\nUsage\n\nlayer = Dense(10, relu)\nlayer = Dense(100, use_bias=false)\n\n\n\n\n\n","category":"type"},{"location":"#Photon.Dropout","page":"Home","title":"Photon.Dropout","text":"Dropout layer with optional the rate (between 0 and 1) of dropout. If no rate is specified, 0.5 (so 50%) will be used.\n\n\n\n\n\n","category":"type"},{"location":"#Photon.Flatten","page":"Home","title":"Photon.Flatten","text":"Flattening Layer. Photon by default already has flattening funcitonality build into the Dense layer, so you won't need to include a separate Flatten layer before a Dense layer.\n\n\n\n\n\n","category":"type"},{"location":"#Photon.HingeLoss","page":"Home","title":"Photon.HingeLoss","text":"Hinge Loss implementation\n\n\n\n\n\n","category":"type"},{"location":"#Photon.MAELoss","page":"Home","title":"Photon.MAELoss","text":"Mean Absolute Error implementation, also referred to as the L1 Loss.\n\n\n\n\n\n","category":"type"},{"location":"#Photon.MSELoss","page":"Home","title":"Photon.MSELoss","text":"Mean Square Error implementation, also referred to as the L2 Loss.\n\nUsage\n\nworkout = Workout(model, MSELoss(), SGD())\n\n\n\n\n\n","category":"type"},{"location":"#Photon.PseudoHuberLoss","page":"Home","title":"Photon.PseudoHuberLoss","text":"Pseudo Huber Loss implementation, somewhere between a L1 and L2 loss.\n\n\n\n\n\n","category":"type"},{"location":"#Photon.Residual","page":"Home","title":"Photon.Residual","text":"Residual Layer. This will stack on the second last dimension. So with and 2D convolution this will be the channel layer (WxHxCxN)\n\n\n\n\n\n","category":"type"},{"location":"#Photon.Sequential","page":"Home","title":"Photon.Sequential","text":"Sequential\n\n\n\n\n\n","category":"type"},{"location":"#Photon.Workout","page":"Home","title":"Photon.Workout","text":"The Workout keeps track of the progress of a training session. At least a model and a loss function needs to be provided. Optional an optimizer and one or more metrics can be provided.\n\nIf no optimizer is provided, SGD will be used. If no metrics are provided only the loss during training and validation will be registered (:loss and :val_loss).\n\nUsage\n\nworkout = Workout(model, mse)\nworkout = Workout(model, nll, SGD())\nworkout = Workout(model, nll, SGD(); acc=BinaryAccuracy())\n\n\n\n\n\n","category":"type"},{"location":"#Photon.autoConvertor-Tuple{Array}","page":"Home","title":"Photon.autoConvertor","text":"autoConvertor converts data to the right format for a model. It uses the context to determine the device (cpu or gpu) and datatype that the data needs to be.\n\nIt supports Tuples, Arrays and KnetArrays and a combination of those.\n\n\n\n\n\n","category":"method"},{"location":"#Photon.back!-Tuple{AutoGrad.Tape,Any}","page":"Home","title":"Photon.back!","text":"Perform the back propagation and update of weights in one go.\n\n\n\n\n\n","category":"method"},{"location":"#Photon.getmetricname","page":"Home","title":"Photon.getmetricname","text":"Function to generate the fully qualified metric name. It uses the metric name and phase (train or valid) to come up with a unique name.\n\n\n\n\n\n","category":"function"},{"location":"#Photon.getmetricvalue","page":"Home","title":"Photon.getmetricvalue","text":"Get the metric value for a fully qualified metric name and a certain step. If step is not provided the last step is used. If no value is found the passed function will not be invoked.\n\nUsage\n\ngetmetricvalue(workout, :val_loss) do value\n    println(\"validation loss\", value)\nend\n\n\n\n\n\n","category":"function"},{"location":"#Photon.gradients","page":"Home","title":"Photon.gradients","text":"Utility function to calculate the gradients. Useful when checking for vanishing or exploding gradients. The returned value is a Vector of (Param, Gradient).\n\n\n\n\n\n","category":"function"},{"location":"#Photon.step!-Tuple{Workout,Any,Any}","page":"Home","title":"Photon.step!","text":"Take a single step in updating the weights of a model. This function will be invoked from fit! to do the actual learning.\n\nFor a minibatch (x,y) of data, the folowing sequence will be executed:\n\nperform the forward pass\ncalculate the loss\nupdate and remember the metrics, if any\ndo the backpropagation and update the weights\n\n\n\n\n\n","category":"method"},{"location":"#Photon.updatemetrics!","page":"Home","title":"Photon.updatemetrics!","text":"Invoke the configured metrics. The loss metric will always be logged and available. Metrics are stored in the history attribute of the workout.\n\n\n\n\n\n","category":"function"},{"location":"#Photon.validate-Tuple{Workout,Any,Any}","page":"Home","title":"Photon.validate","text":"Validate a minibatch and calculate the loss and metrics. Typically this function is called from the fit! method.\n\n\n\n\n\n","category":"method"},{"location":"#Serialization.serialize-Tuple{Serialization.AbstractSerializer,Knet.KnetArray}","page":"Home","title":"Serialization.serialize","text":"Enable saving and loading of models by specialized KnetArray methods for Julia serialization This will effectively move a GPU weight to the CPU before serialing it and move it back to the GPU when deserializing.\n\n\n\n\n\n","category":"method"},{"location":"#Photon.Context","page":"Home","title":"Photon.Context","text":"Context is used by various parts of Photon to determine what the device and datatype should be for Arrays. It also allows to quickly switch between GPU and CPU based models.\n\nAttributes\n\ndevice::Symbol the type of device. For now supported :cpu and :gpu\ndeviceId::Int the id of the device, useful for example if you multiple GPU's\ndtype::Type the type of data you want to use. Most common are Float32, Float16 or Float64.\n\n\n\n\n\n","category":"type"},{"location":"#Photon.Conv","page":"Home","title":"Photon.Conv","text":"Convolutional layer that serves as the base for Conv2D and Conv3D\n\n\n\n\n\n","category":"type"},{"location":"#Photon.ConvTranspose","page":"Home","title":"Photon.ConvTranspose","text":"ConvTranspose layer\n\n\n\n\n\n","category":"type"},{"location":"#Photon.Loss","page":"Home","title":"Photon.Loss","text":"Base type for the loss functions. However Photon accepts any functoon as a loss function as long as it is callable and returns the loss as a scalar value.\n\n fn(y_pred, y_true) :: Number\n\n\n\n\n\n","category":"type"},{"location":"#Photon.Meter","page":"Home","title":"Photon.Meter","text":"A meter is reponsible for presenting metric values. This can be printing it to the console output, showing it on TensorBoard of storing it in a database.\n\n\n\n\n\n","category":"type"},{"location":"#Photon.PoolingLayer","page":"Home","title":"Photon.PoolingLayer","text":"Pooling layers\n\n\n\n\n\n","category":"type"},{"location":"#Photon.StackedLayer","page":"Home","title":"Photon.StackedLayer","text":"Common behavior for stacked layers that enables to access them as arrays\n\n\n\n\n\n","category":"type"}]
}
