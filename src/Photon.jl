
module Photon

using Knet

include("layers/core.jl")
include("layers/conv.jl")
include("layers/recurrent.jl")
include("layers/container.jl")
include("utils.jl")


export Layer, LazyLayer, Dense, BatchNorm, RNN_RELU, RNN_TANH,
	  Conv2D, Dropout, Sequential, Flatten, MaxPool2D, AvgPool2D,
	  LSTM, GRU, Residual, Concurrent, AdaptiveAvgPool, AdaptiveMaxPool,
	  Activation, add, forward, ctx, ContextSwitch

@info "Loaded Photon"

end
