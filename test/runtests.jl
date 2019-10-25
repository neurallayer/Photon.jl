using Photon
using Knet

## Simple conv model
using Test

@testset "Photon" begin

@info "Running Unit tests"

@info "Testing Basics"
include("./basic.jl")

@info "Testing Large models"
include("./modelzoo.jl")

@info "Testing training of models"
include("./training.jl")

end
