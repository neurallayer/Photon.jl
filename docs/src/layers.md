## Basic

```@docs
Dense
Dropout
BatchNorm
Flatten
```

## Container
Containers are special type of layers that contain other layers.

```@docs
Sequential
Concurrent
Residual
```

## Convolutional
Convolution layer for 2D and 3D spatial data. A 2D convolutional layer requires
a 4D Array in the shape of WxHxCxN (width x height x channels x batch).

```@docs
Conv2D
Conv2DTranspose

Conv3D
Conv3DTranspose
```

## Pooling

```@docs
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
