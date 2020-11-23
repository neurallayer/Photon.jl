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
include("optimisers/optimisers.jl")
include("train.jl")
include("callbacks/callbacks.jl")
include("data/data.jl")
include("zoo/zoo.jl")


@debug "Loaded Photon"

end
