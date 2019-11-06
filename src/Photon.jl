
module Photon

using Statistics, Printf, Serialization
import Knet

include("core.jl")
include("layers/core.jl")
include("layers/conv.jl")
include("layers/recurrent.jl")
include("layers/container.jl")
include("optimizers/optimise.jl")
include("train.jl")
include("losses.jl")
include("metrics/core.jl")
include("metrics/meters.jl")
include("utils.jl")

export Layer, LazyLayer, Dense, BatchNorm, RNN, Conv2DTranspose,
	  Conv2D, Conv3D, output_size, Dropout, Sequential, Flatten, MaxPool2D, AvgPool2D,
	  LSTM, GRU, Residual, Concurrent, AdaptiveAvgPool, AdaptiveMaxPool,
	  Activation, add, forward, ContextSwitch, ADAM


dir(path...) = joinpath(dirname(@__DIR__),path...)

@info "Loaded Photon"

end
