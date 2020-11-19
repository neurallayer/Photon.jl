
module Layers

using Knet
using Photon: Tensor, Shape

include("core.jl")
include("conv.jl")
include("recurrent.jl")
include("container.jl")

end