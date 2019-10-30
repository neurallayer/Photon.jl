

export getContext, setContext, resetContext, ctx, hasgpu, is_on_gpu, KorA

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


function is_on_gpu()
	ctx.device == :gpu
end

hasgpu() = Knet.gpu() >= 0

function getContext()
  ctx
end

function setContext(;device=ctx.device, deviceId=ctx.deviceId, dtype=ctx.dtype)
  ctx.device = device
  ctx.deviceId = deviceId
  ctx.dtype= dtype
  getContext()
end


function resetContext()
	global ctx
	ctx = Context()
end

"""
KorA makes it easy to move an array to the GPU or the other way around
"""
function KorA(arr::Array)
    (ctx.device == :gpu) ? Knet.KnetArray(arr) : arr
end

function KorA(arr::Knet.KnetArray)
    (ctx.device == :cpu) ? Array(arr) : arr
end

function KorA(arr::Tuple)
    (KorA(elem) for elem in arr)
end


addlast(x) = reshape(x, (size(x)...,1))

droplast(x) = reshape(x, (size(x)[1:end-1]...))
