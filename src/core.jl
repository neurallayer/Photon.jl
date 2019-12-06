

export Loss, Context, getContext, setContext, resetContext, hasgpu, is_on_gpu, KorA

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

const Tensor{T,N} = Union{
						AbstractArray{T,N},
						Knet.KnetArray{T,N},
						Knet.Value{Array{T,N}},
						Knet.Value{Knet.KnetArray{T,N}}}

const Tensors = Union{Tensor, Tuple{Tensor}}


"""
Context is used by various parts of Photon to determine what the device and
datatype should be for data. It allows for quickly switches between GPU and CPU
based environments.

# Attributes
- device::Symbol the type of device. For now supported :cpu and :gpu
- deviceId::Int the id of the device, useful for example if you multiple GPU's
- dtype::Type the type you want to use for parameters. Most common are
  Float32, Float16 or Float64.
"""
mutable struct Context
	device::Symbol
	deviceId::Int
	dtype::Type

	function Context()
		device = Knet.gpu() >= 0 ? :gpu : :cpu
		new(device, 0, Float32)
	end
end

global ctx = Context()


function is_on_gpu()::Bool
	ctx.device == :gpu
end

hasgpu() = Knet.gpu() >= 0

"""
Get the currently configured Context.
"""
function getContext()::Context
  ctx
end

"""
Update the Context with the provided values. If values are no specified, the
current value will be used.
"""
function setContext(;device=ctx.device, deviceId=ctx.deviceId, dtype=ctx.dtype)
  ctx.device = device
  ctx.deviceId = deviceId
  ctx.dtype= dtype
  getContext()
end

"""
Reset the Context to its default values. That means if there is a GPU detected
GPU, otherwise CPU. And as a datatype Float32.
"""
function resetContext()
	global ctx
	ctx = Context()
end

addlast(x) = reshape(x, (size(x)...,1))
droplast(x) = reshape(x, (size(x)[1:end-1]...))

"""
Mover converts data to the right device liek a CPU or GPU. However implememetations
can provide extra functionality like also taking care of the correct data types.

The default Mover is SmartMover.isconcretetype
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
	move_float

	SmartMover(move_float=true) = new(move_float)
end

function (m::SmartMover)(arr::Array)
	if m.move_float && (eltype(arr) isa AbstractFloat)
		arr = convert(Array{ctx.dtype}, arr)
	end
	ctx.device == :gpu ? Knet.KnetArray(arr) : arr
end

function (m::SmartMover)(arr::Knet.KnetArray)
	if m.move_float && (eltype(arr) isa AbstractFloat)
		arr = convert(Knet.KnetArray{ctx.dtype}, arr)
	end
	ctx.device == :gpu ? arr : Array(arr)
end

(m::SmartMover)(t::Tuple)= (m(elem) for elem in t)


"""
KorA is just an instance of SmartMover.
"""
KorA = SmartMover()


# small util
makeArray(x::AbstractArray) = x
makeArray(x) = Vector(x)
