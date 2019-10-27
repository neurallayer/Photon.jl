

export getContext, setContext, ctx

mutable struct Context
	devType::Symbol
	devId::Int
	dataType::Type

	function Context()
		devType = gpu() >= 0 ? :gpu : :cpu
		new(devType, 0, Float32)
	end
end

global ctx = Context()


function is_on_gpu()
	ctx.devType == :gpu
end



function getContext()
  ctx
end

function setContext(;device=ctx.devType, deviceId=ctx.devId, dtype=ctx.dataType)
  ctx.devType = device
  ctx.devId = deviceId
  ctx.dataType= dtype
  getctx()
end
