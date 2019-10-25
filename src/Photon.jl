
module Photon

using Knet
using Statistics

include("layers/core.jl")
include("layers/conv.jl")
include("layers/recurrent.jl")
include("layers/container.jl")
include("utils.jl")
include("optimise.jl")
include("train.jl")
include("losses.jl")


export Layer, LazyLayer, Dense, BatchNorm, RNN_RELU, RNN_TANH,
	  Conv2D, Dropout, Sequential, Flatten, MaxPool2D, AvgPool2D,
	  LSTM, GRU, Residual, Concurrent, AdaptiveAvgPool, AdaptiveMaxPool,
	  Activation, add, forward, ctx, ContextSwitch, ADAM, Workout, fit!

@info "Loaded Photon"

end
