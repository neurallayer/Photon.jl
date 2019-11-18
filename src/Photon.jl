
module Photon

using Statistics, Reexport
import Knet

include("core.jl")
include("layers/core.jl")
include("layers/conv.jl")
include("layers/recurrent.jl")
include("layers/container.jl")
include("train.jl")
include("losses.jl")

include("metrics/Metrics.jl")
@reexport using .Metrics

include("data/Data.jl")
@reexport using .Data

include("utils.jl")

export Dense, BatchNorm, Sequential, Flatten, Residual, Concurrent, Dropout,
	   Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose, output_size,
	   MaxPool2D, AvgPool2D, AdaptiveAvgPool, AdaptiveMaxPool,
	   RNN, LSTM, GRU,
	   Activation, add, forward, ContextSwitch

dir(path...) = joinpath(dirname(@__DIR__),path...)

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
