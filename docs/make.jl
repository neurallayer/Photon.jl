using Pkg;
Pkg.activate(joinpath(@__DIR__, "..")); Pkg.instantiate()
Pkg.activate(); Pkg.instantiate()

pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Documenter, Photon

makedocs(modules=[Flux, NNlib],
         sitename = "Photon",
         pages = ["Home" => "index.md",
                  "Community" => "community.md"],
         format = Documenter.HTML())

deploydocs(repo = "github.com/neurallayer/Photon.jl.git")
