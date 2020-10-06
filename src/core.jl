

export Loss, KorA

const ϵ = 10e-8

abstract type MetricStore end
abstract type Layer end
abstract type Metric end
abstract type Optimizer end

"""
Abstract type for the loss functions. However Photon accepts
any function as a loss function as long as it is callable and
returns the loss value as a scalar type.
"""
abstract type Loss end

const Shape = Tuple{Vararg{Int}}

"""
const Tensor{T,N} = Union{
						AbstractArray{T,N},
						Knet.KnetArray{T,N},
						Knet.Value{Array{T,N}},
						Knet.Value{Knet.KnetArray{T,N}}}
						
const Tensors = Union{Tensor, Tuple{Tensor}}
"""

const Tensor = Any

hasgpu() = CUDA.functional()
addlast(x) = reshape(x, (size(x)...,1))
droplast(x) = reshape(x, (size(x)[1:end-1]...))

"""
Moves data to the right device like a CPU or GPU. However implememetations
can provide extra functionality like also taking care of the correct data types.

The default Mover used by Photon is SmartMover if no other mover is specified during
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
