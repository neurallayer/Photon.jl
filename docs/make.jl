using Pkg;
Pkg.activate(joinpath(@__DIR__, "..")); Pkg.instantiate()
Pkg.activate(); Pkg.instantiate()

pushfirst!(LOAD_PATH, joinpath(@__DIR__, ".."))

using Documenter, Photon

makedocs(modules=[Photon],
         sitename = "Photon",
         pages = ["Get Started" => "index.md",
                  "API" => [
                        "Core"    => "core.md",
                        "Layers"  => "layers.md",
                        "Losses"  => "losses.md",
                        "Metrics" => "metrics.md",
                        "Data"    => "data.md"
                  ],
                  "Community" => "community.md"],
         format = Documenter.HTML(
                  prettyurls = get(ENV, "CI", nothing) == "true"
         ))

deploydocs(
        repo = "github.com/neurallayer/Photon.jl.git",
        target = "build")
