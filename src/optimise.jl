mutable struct Descent
  eta::Float64
end

Descent() = Descent(0.1)

function apply!(o::Descent, x, Δ)
  Δ .*= o.eta
end



mutable struct Momentum2
  eta::Float64
  rho::Float64
  velocity::IdDict
end


function Momentum2(η = 0.01, ρ = 0.9)
    Momentum2(η, ρ, IdDict())
end

function update!(o::Momentum2, params, zerograd=true)
    η, ρ = o.eta, o.rho
    for x in params
        Δ = x.opt == nothing ? continue : x.opt
        v = get(o.velocity,x) do; zero(x) end
        @. v = ρ * v - η * Δ
        o.velocity[x] = v

        x.value .+= v
        zerograd && (x.opt = nothing)
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

ϵ = 10e-8

mutable struct ADAM
  eta::Float64
  beta::Tuple{Float64,Float64}
  state::IdDict
end

ADAM(η = 0.001, β = (0.9, 0.999)) = ADAM(η, β, IdDict())

function update!(o::ADAM, params, zerograd=true)
  η, β = o.eta, o.beta
  for x in params
    Δ = x.opt == nothing ? continue : x.opt
    mt, vt, βp = get(o.state, x) do; (zero(x), zero(x), β) end
    @. mt = β[1] * mt + (1 - β[1]) * Δ
    @. vt = β[2] * vt + (1 - β[2]) * Δ^2
    @. Δ =  mt / (1 - βp[1]) / (√(vt / (1 - βp[2])) + ϵ) * η
    o.state[x] = (mt, vt, βp .* β)

    x.value .-= Δ
    zerograd && (x.opt = nothing)
  end

end

@info "Loaded Optimisers modules"
