
module Photon

using Printf
using Statistics
using Reexport
import Knet
import CUDA

include("core.jl")
include("layers/Layers.jl")
include("train.jl")
include("losses.jl")

include("callbacks.jl")
export Callback, AutoSave, EpochSave, EarlyStop
export Meter, ConsoleMeter, SilentMeter, TensorBoardMeter, FileMeter, PlotMeter

include("metrics.jl")
export getmetricname, SmartReducer, update!, history, BinaryAccuracy, OneHotBinaryAccuracy


include("data/Data.jl")
@reexport using .Data

include("utils.jl")

export Dense, BatchNorm, Flatten, Dropout, hasgpu,
	   StackedLayer, Sequential, Residual, Concurrent,
	   Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose, output_size,
	   PoolingLayer, MaxPool2D, AvgPool2D, MaxPool3D, AvgPool3D,
	   AdaptiveAvgPool, AdaptiveMaxPool,
	   RNN, LSTM, GRU,
	   Activation, add,
	   Mover, SmartMover, batchfirst, batchlast

dir(path...) = joinpath(dirname(@__DIR__), path...)

# Re-export some of the Knet features
module K
	using Knet
	export relu, elu, selu, sigm
	export softmax, nll
	export Adam, SGD, Momentum, Nesterov, Adagrad, Rmsprop, Adadelta
	export xavier, gaussian, bilinear
end

using .K

export relu, elu, selu, sigm
export softmax, nll
export Adam, SGD, Momentum, Nesterov, Adagrad, Rmsprop, Adadelta
export xavier, gaussian, bilinear

@debug "Loaded Photon"

end
