
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

export Dense, BatchNorm, RNN, Conv2DTranspose,
	  Conv2D, Conv3D, output_size, Dropout, Sequential, Flatten, MaxPool2D, AvgPool2D,
	  LSTM, GRU, Residual, Concurrent, AdaptiveAvgPool, AdaptiveMaxPool,
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
