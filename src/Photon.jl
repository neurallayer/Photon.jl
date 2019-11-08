
module Photon

using Statistics, Printf
import Knet

include("core.jl")
include("layers/core.jl")
include("layers/conv.jl")
include("layers/recurrent.jl")
include("layers/container.jl")
include("train.jl")
include("losses.jl")
include("metrics/core.jl")
include("metrics/meters.jl")
include("utils.jl")

export Dense, BatchNorm, RNN, Conv2DTranspose,
	  Conv2D, Conv3D, output_size, Dropout, Sequential, Flatten, MaxPool2D, AvgPool2D,
	  LSTM, GRU, Residual, Concurrent, AdaptiveAvgPool, AdaptiveMaxPool,
	  Activation, add, forward, ContextSwitch


dir(path...) = joinpath(dirname(@__DIR__),path...)

# Rexport some of the Knet features
module K
	using Knet
	export relu, softmax, nll
	export Adam, SGD, Momentum
	export xavier
end

using .K

export relu, softmax, nll
export Adam, SGD, Momentum
export xavier


@info "Loaded Photon"

end
