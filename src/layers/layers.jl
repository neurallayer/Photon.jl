
module Layers

using Knet
using Photon: Tensor, Shape

include("core.jl")
include("conv.jl")
include("recurrent.jl")
include("container.jl")

export Dense, Sequential, Flatten, Activation, BatchNorm, Dropout, Concurrent, Softmax, LogSoftmax,
    Conv1D, Conv2D, Conv3D,  Conv2DTranspose, MaxPool2D, AvgPool2D, MaxPool2D, AdaptiveAvgPool, AdaptiveMaxPool, 
    RNN, LSTM, GRU, add, output_size,
    relu, elu, selu, sigm # some Knet activation functions

end