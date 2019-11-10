
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
include("data/dataloader.jl")
include("data/dataset.jl")
include("utils.jl")

export Dense, BatchNorm, RNN, Conv2DTranspose,
	  Conv2D, Conv3D, output_size, Dropout, Sequential, Flatten, MaxPool2D, AvgPool2D,
	  LSTM, GRU, Residual, Concurrent, AdaptiveAvgPool, AdaptiveMaxPool,
	  Activation, add, forward, ContextSwitch,
	  Dataloader, Dataset, TestDataset, ImageDataset, ArrayDataset, JLDDataset


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


@info "Loaded Photon"

end
