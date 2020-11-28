

export KorA # SmartMover

const ϵ = 10e-8

abstract type MetricStore end

abstract type Optimizer end


const Shape = Tuple{Vararg{Int}}

"""
const SingleTensor{T,N} = Union{
						AbstractArray{T,N},
						Knet.KnetArray{T,N},
						Knet.Value{Array{T,N}},
						Knet.Value{Knet.KnetArray{T,N}}}
						
const Tensor = Union{SingleTensor, Tuple{Tensor}}

Due to many possiblities we just use Tensor type to show intend but not 
using the above definition.
"""
const Tensor = Any

hasgpu() = CUDA.functional()
addlast(x) = reshape(x, (size(x)...,1))
droplast(x) = reshape(x, (size(x)[1:end-1]...))
"""
Move the batch from the first to the last dimension
"""
function batchlast(a::AbstractArray)
      p = collect(1:length(size(a)))
      push!(p, popfirst!(p))
      permutedims(a,p)
end


"""
Move the batch from the last to the first dimension
"""
function batchfirst(a::AbstractArray)
      p = collect(1:length(size(a)))
      pushfirst!(p, pop!(p))
      permutedims(a,p)
end




"""
Moves data to the right device like a CPU or GPU. However implememetations
can provide extra functionality like also taking care of the correct data types.

The default Mover used by Photon is the SmartMover if no other mover is specified during
instantiation of the Workout.
"""
abstract type Mover end

"""
SmartMover converts data to the right device and optional data type for a model.
It uses the context to determine the device (cpu or gpu) and datatype
that the data needs to be.

It move_float is true, SmartMover will ensure that any provided Array of the type
AbstractFloat will convert to the Float type as defined in the context.

It supports Tuples, Arrays and KnetArrays and a combination of those.
"""
struct SmartMover <: Mover
	atype::Type

	function SmartMover(atype=nothing)
		if isnothing(atype)
			if hasgpu() 
				atype = Knet.KnetArray{Float32}
			else
				atype = Array{Float32}
			end
		end
		new(atype)
	end
end

function (m::SmartMover)(array)
	convert(m.atype, array)
end

(m::SmartMover)(t::Tuple)= (m(elem) for elem in t)


"""
KorA is just an instance of SmartMover.
"""
KorA = SmartMover()


# small util
makeArray(x::AbstractArray) = x
makeArray(x) = Vector(x)



"""
Stores the calculated metrics. If multiple values are provided at the same step
(like is the case with validation metrics), the moving average over those values
will be stored instead.
"""
struct SmartReducer <: MetricStore
    state::Dict{Int, Real}
    momentum::Real
    SmartReducer(momentum=0.9) = new(Dict(), momentum)
end

function update!(r::SmartReducer, step::Int, value::Real)
    if haskey(r.state, step)
        r.state[step] = r.momentum * r.state[step] + (1-r.momentum) * value
    else
        r.state[step] = value
    end
end


"""
Function to generate the fully qualified metric name. It uses the metric name
and the phase (:train or :valid) to come up with a unique name.

```julia
getmetricname(:loss, :train) # return is :loss
getmetricname(:loss, :valid) # return is :val_loss
```
"""
function getmetricname(metric::Symbol, phase=:train)::Symbol
    metricname = phase == :train ? metric : Symbol("val_", metric)
end


