
## Metrics
Metrics can be passed to the Workout initializer and provide extra insights next
to the loss value. Every metric is invoked with two parameters: predicted outcome
and the labels.

```@docs
BinaryAccuracy
OneHotBinaryAccuracy
SmartReducer
```


## Callbacks
Callbacks can be passed to the train! function and provide a way to extend the functionality
of Photon. Their functionality ranges from callbacks that generate output (like console or plots),
to callbacks that save the model and callbacks that abort the training due to lack of progress.

```@docs
Meter
ConsoleMeter
SilentMeter
FileMeter
PlotMeter
TensorBoardMeter
EarlyStop
AutoSave
EpochSave
```


## Utils
Photon comes with a number of utility functions for common tasks with regards
to using metrics.

```@docs
getmetricname
history
plotmetrics
```
