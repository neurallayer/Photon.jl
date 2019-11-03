

export SGD, ADAM, Momentum


mutable struct SGD <: Optimizer
  eta::Float64

  SGD(lr=0.01) = new(lr)
end


function update!(o::SGD, params)
    for x in params
        Δ = x.opt == nothing ? continue : x.opt
        x.value .-=  Δ .* o.eta
    end
end


mutable struct Momentum <: Optimizer
  eta::Float64
  rho::Float64
  velocity::IdDict

  Momentum(η = 0.01, ρ = 0.9) = new(η, ρ, IdDict())
end

function update!(o::Momentum, params)
    η, ρ = o.eta, o.rho
    for x in params
        Δ = x.opt == nothing ? continue : x.opt
        v = get(o.velocity,x) do; zero(x) end
        @. v = ρ * v - η * Δ
        o.velocity[x] = v

        x.value .+= v
    end
end


"""
    ADAM(η, β::Tuple)
Implements the ADAM optimiser.
## Paramters
  - Learning Rate (`η`): Defaults to `0.001`.
  - Beta (`β::Tuple`): The first element refers to β1 and the second to β2. Defaults to `(0.9, 0.999)`.
## Examples
```julia
opt = ADAM() # uses the default η = 0.001 and β = (0.9, 0.999)
opt = ADAM(0.001, (0.9, 0.8))
```
## References
[ADAM](https://arxiv.org/abs/1412.6980v8) optimiser.
"""



mutable struct ADAM <: Optimizer
  eta::Float64
  beta::Tuple{Float64,Float64}
  state::IdDict
end

ADAM(η = 0.001, β = (0.9, 0.999)) = ADAM(η, β, IdDict())

function update!(o::ADAM, params)
  η, β = o.eta, o.beta
  for x in params
    Δ = x.opt == nothing ? continue : x.opt
    mt, vt, βp = get(o.state, x) do; (zero(x), zero(x), β) end
    @. mt = β[1] * mt + (1 - β[1]) * Δ
    @. vt = β[2] * vt + (1 - β[2]) * Δ^2
    @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) * η
    o.state[x] = (mt, vt, βp .* β)

    x.value .-= Δ
  end

end

@info "Loaded Optimisers modules"
