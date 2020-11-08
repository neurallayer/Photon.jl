module PhotonTest

@info "Running Unit tests"

using Photon
using Test

@testset "Photon" begin

@info "Testing Basics"
include("./basic.jl")

@info "Testing Dense Layers"
include("./dense.jl")

@info "Testing Convolutional Layers"
include("./conv.jl")

@info "Testing Recurrent Layers"
include("./recurrent.jl")

@info "Testing Large models"
include("./modelzoo.jl")

@info "Testing training of models"
include("./training.jl")
include("./optimise.jl")
include("./loss.jl")

@info "Testing metrics and meters"
include("./metrics.jl")

@info "Testing complex training scenarios"
include("./complex.jl")

@info "Testing dataset and dataloader"
include("./data.jl")

end

end
