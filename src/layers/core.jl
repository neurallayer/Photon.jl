

function getparam(d...;init=Knet.xavier)
	et = ctx.dtype
	atype = ctx.device == :gpu ? Knet.KnetArray{et} : Array{et}
	Knet.Param(atype(init(d...)))
end


const _layernames = Dict{String,Int}()

function get_layername(layername; kwargs...)
	info = Dict(kwargs)
	if haskey(info, :name)
		info[:name]
	else
		if ! haskey(_layernames, layername)
			_layernames[layername] = 1
		end
		id = _layernames[layername]
		_layernames[layername] += 1
		"$(layername)_$(id)"
	end
end


abstract type Layer end
abstract type LazyLayer <: Layer end

function (layer::Layer)(X)
	@debug "Calling $(typeof(layer))" size(X)
    call(layer, X)
end


## Generic code
function (layer::LazyLayer)(X)
	@assert hasproperty(layer, :built) "LazyLayer $layer without built property detected"
	@assert isa(layer.built, Bool) "LazyLayer $layer built property not a boolean"

    if ! layer.built
		@debug "Initializing $(typeof(layer))" size(X) layer
        build(layer, size(X)[1:end-1])
		layer.built = true
    end
	@debug "Calling $(typeof(layer))" size(X)
    call(layer, X)
end




"""
Fully connected layer with an optinal bias
"""
mutable struct Dense <:LazyLayer
    units::Int
	activation::Function
	use_bias::Bool
	name::String
	built::Bool
	params::NamedTuple

	function Dense(units::Int, activation=identity; use_bias=true, kwargs...)
		@assert units > 0
		name = get_layername("dense"; kwargs...)
	    new(units, activation, use_bias, name, false, (w=nothing, b=nothing))
	end
end

function call(layer::Dense, X)
	X = Knet.mat(X) # Flatten if required

	w,b = layer.params
	if layer.use_bias
    	layer.activation.(w*X .+ b)
	else
		layer.activation.(w*X)
	end
end

function build(layer::Dense, shape::Tuple)
	nInput = length(shape) > 1 ? *(shape...) : shape[1]

    w = getparam(layer.units, nInput)
	b = nothing
    if layer.use_bias
		b = getparam(layer.units, init=zeros)
	end
	layer.params = (w=w,b=b)
end



"""
Flattening Layer
"""
mutable struct Flatten <: Layer
	dims
	Flatten(dims=nothing) = new(dims)
end

function call(layer::Flatten, X)
	layer.dims == nothing ? Knet.mat(X) : Knet.mat(X, dims=layer.dims)
end



## Activation
mutable struct Activation <:Layer
	activation
end

function call(c::Activation, x)
	c.activation.(x)
end


"""
Dropout layer
"""
struct Dropout <: Layer
	rate

	function Dropout(rate=0.5)
		@assert rate <= 1.0
		@assert rate >= 0.0
		new(rate)
	end
end

function call(d::Dropout,X)
	Knet.dropout(X, d.rate)
end

## Batch Normalization Layer
struct BatchNorm <: Layer

  	moments

	function BatchNorm()
		new(Knet.bnmoments())
	end
end

function call(bn::BatchNorm, X)
	Knet.batchnorm(X, bn.moments)
end



## SwitchContext

"""
Beginning of allowing for a single model instance to run on multiple devices
(expiremental)
"""
struct ContextSwitch <: Layer

  	device
	deviceId
	dtype

	function ContextSwitch(;device=ctx.device,deviceId=ctx.deviceId,dtype=ctx.dtype)
		new(device,deviceId,dtype)
	end
end

function call(c::ContextSwitch, X)
	setContext(device=c.device, deviceId=c.deviceId, dtype=c.dtype)

	if c.device == :cpu
		X = convert(Array{c.dtype},X)
	elseif c.device == :gpu
		X = convert(Knet.KnetArray{c.dtype},X)
	end
	X
end



@info "Loaded Core modules"
