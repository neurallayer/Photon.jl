
module Photon

using Printf
using Statistics
import Knet
import CUDA

include("core.jl")
include("layers/Layers.jl")
include("train.jl")
include("losses.jl")
include("callbacks.jl")
include("metrics.jl")
include("data/Data.jl")
include("utils.jl")
include("zoo/Zoo.jl")

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
