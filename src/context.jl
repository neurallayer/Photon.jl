export Context, getContext, setContext, resetContext, hasgpu, is_on_gpu, KorA

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

global const __ctx = Context()


function is_on_gpu()::Bool
	__ctx.device == :gpu
end

hasgpu() = Knet.gpu() >= 0

"""
Get the currently configured Context.
"""
function getContext()::Context
  __ctx
end

"""
Update the Context with the provided values. If values are no specified, the
current value will be used.
"""
function setContext(;device=nothing, deviceId=nothing, dtype=nothing)
  if device !== nothing __ctx.device = device end
  if deviceId !== nothing __ctx.deviceId = deviceId end
  if dtype !== nothing __ctx.dtype= dtype end
end

"""
Reset the Context to its default values. That means if there is a GPU detected
GPU, otherwise CPU. And as a datatype Float32.
"""
function resetContext()
	__ctx.device = Knet.gpu() >= 0 ? :gpu : :cpu
	__ctx.deviceId = 0
	__ctx.dtype = Float32
end
