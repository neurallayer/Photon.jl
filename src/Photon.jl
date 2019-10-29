
module Photon

using Statistics
import Knet

include("core.jl")
include("utils.jl")
include("layers/core.jl")
include("layers/conv.jl")
include("layers/recurrent.jl")
include("layers/container.jl")
include("optimizers/optimise.jl")
include("train.jl")
include("losses.jl")
include("metrics/core.jl")

export Layer, LazyLayer, Dense, BatchNorm, RNN_RELU, RNN_TANH,
	  Conv2D, Dropout, Sequential, Flatten, MaxPool2D, AvgPool2D,
	  LSTM, GRU, Residual, Concurrent, AdaptiveAvgPool, AdaptiveMaxPool,
	  Activation, add, forward, ContextSwitch, ADAM


dir(path...) = joinpath(dirname(@__DIR__),path...)

@info "Loaded Photon"

end
