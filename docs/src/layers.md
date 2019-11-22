## Basic


```@docs
Dense
Dropout
BatchNorm
Flatten
```

## Container
Containers are special type of layers that contain other layers. They typically
extend the abstract type that allows to use them as regular Vectors.

```@docs
StackedLayer
Sequential
Concurrent
Residual
```

## Convolutional
Photon contains convolutional layers for 2D and 3D spatial data.

A 2D convolutional layer would require a 4D Array in the shape of WxHxCxN
(width x height x channels x batch). So for a typical image classification
problem this could look like: 224 x 224 x 3 x 8 (224 by 224 image, with 3 colors
and 8 samples per batch).


```@docs
Conv2D
Conv2DTranspose

Conv3D
Conv3DTranspose
```

## Pooling

```@docs
PoolingLayer
MaxPool2D
AvgPool2D
MaxPool3D
AvgPool3D
AdaptiveAvgPool
AdaptiveMaxPool
```

## Recurrent


```@docs
RNN
LSTM
GRU
```

## Experimental

```@docs
ContextSwitch
```
