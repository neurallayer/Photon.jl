

export getContext, setContext, ctx, hasgpu, KorA

mutable struct Context
	devType::Symbol
	devId::Int
	dataType::Type

	function Context()
		devType = Knet.gpu() >= 0 ? :gpu : :cpu
		new(devType, 0, Float32)
	end
end

global ctx = Context()


function is_on_gpu()
	ctx.devType == :gpu
end

hasgpu() = Knet.gpu() >= 0

function getContext()
  ctx
end

function setContext(;device=ctx.devType, deviceId=ctx.devId, dtype=ctx.dataType)
  ctx.devType = device
  ctx.devId = deviceId
  ctx.dataType= dtype
  getctx()
end

"""
KorA makes it easy to move an array to the GPU or the other way around
"""
function KorA(arr::Array)
    (ctx.devType == :gpu) ? Knet.KnetArray(arr) : arr
end

function KorA(arr::Knet.KnetArray)
    (ctx.devType == :cpu) ? Array(arr) : arr
end

function KorA(arr::Tuple)
    (KorA(elem) for elem in arr)
end


addlast(x) = reshape(x, (size(x)...,1))

droplast(x) = reshape(x, (size(x)[1:end-1]...))
