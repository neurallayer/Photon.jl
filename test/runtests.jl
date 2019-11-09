@info "Running Unit tests"

using Photon
using Test

@testset "Photon" begin

@info "Testing Basics"
include("./basic.jl")
include("./dense.jl")
include("./conv.jl")
include("./recurrent.jl")

@info "Testing Large models"
include("./modelzoo.jl")

@info "Testing training of models"
include("./training.jl")
include("./optimise.jl")
include("./loss.jl")

@info "Testing complex training scenarios"
include("./complex.jl")

@info "Testing dataset and dataloader"
include("./data.jl")

end
