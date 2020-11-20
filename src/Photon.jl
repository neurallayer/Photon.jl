__precompile__(true)


module Photon

using Printf
using Statistics
import Knet
import CUDA

include("core.jl")
include("layers/layers.jl")
include("metrics/metrics.jl")
include("losses/losses.jl")
include("train.jl")
include("callbacks/callbacks.jl")
include("data/data.jl")
include("utils.jl")
include("zoo/zoo.jl")

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


@debug "Loaded Photon"

end
