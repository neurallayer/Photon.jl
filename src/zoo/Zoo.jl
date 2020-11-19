module Zoo

using Photon
using Knet
using Photon.Layers
import Photon: hasgpu, KorA

include("alexnet.jl")
include("densenet.jl")
include("vgg.jl")

end
