

function getparam(d...;init=Knet.xavier)
	ctx = getcontext()
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



abstract type LazyLayer <: Layer end

## Generic code, X can be a Tensor or a Tuple
function (layer::Layer)(X)
	@debug "Calling $(typeof(layer))" layer
    call(layer, X)
end


## Generic code, X can be a Tensor or a Tuple
function (layer::LazyLayer)(X)
	@assert hasproperty(layer, :built) "LazyLayer $layer without built property detected"
	@assert isa(layer.built, Bool) "LazyLayer $layer built property not a boolean"

    if ! layer.built
		@debug "Building $(typeof(layer))" layer
        build(layer, size(X)[1:end-1])
		layer.built = true
    end
	@debug "Calling $(typeof(layer))" layer
    call(layer, X)
end


"""
Regular densely-connected NN layer.

Dense implements the operation: output = activation(dot(input, weight) + bias) where
activation is the element-wise activation function passed as the activation argument,
weight is a weights matrix created by the layer, and bias is a bias vector created
by the layer (only applicable if use_bias is true).

# Usage

```julia
layer = Dense(10, relu)
layer = Dense(100, use_bias=false)
```

"""
mutable struct Dense <:LazyLayer
    units::Int
	activation::Function
	use_bias::Bool
	name::String
	built::Bool
	init::NamedTuple
	params::NamedTuple

	function Dense(units::Int, activation=identity; use_bias=true,
		initw = Knet.xavier, initb = zeros, kwargs...)
		@assert units > 0 "Units of a Dense layer should be > 0"
		name = get_layername("dense"; kwargs...)
	    new(units, activation, use_bias, name, false,
		(w=initw, b=initb), (w=nothing, b=nothing))
	end
end

function call(layer::Dense, X::Tensor)
	X = Knet.mat(X) # Flatten if required

	w,b = layer.params
	if layer.use_bias
    	layer.activation.(w*X .+ b)
	else
		layer.activation.(w*X)
	end
end

function build(layer::Dense, shape::Shape)
	# nInput = length(shape) > 1 ? *(shape...) : shape[1]
	nInput = *(shape...)

    w = getparam(layer.units, nInput, init=layer.init.w)
	b = nothing
    if layer.use_bias
		b = getparam(layer.units, init=layer.init.b)
	end
	layer.params = (w=w,b=b)
end


"""
Flattening Layer. Photon by default already has flattening funcitonality
build into the Dense layer, so you won't need to include a separate Flatten
layer before a Dense layer.
"""
mutable struct Flatten <: Layer
	dims
	Flatten(dims=nothing) = new(dims)
end

function call(layer::Flatten, X::Tensor)
	layer.dims === nothing ? Knet.mat(X) : Knet.mat(X, dims=layer.dims)
end


## Activation
mutable struct Activation <:Layer
	activation
end

function call(c::Activation, X::Tensor)
	c.activation.(X)
end


"""
Dropout layer with optional the rate (between 0 and 1) of dropout. If
no rate is specified, 0.5 (so 50%) will be used.
"""
struct Dropout <: Layer
	rate::Float64

	function Dropout(rate=0.5)
		@assert rate <= 1.0
		@assert rate >= 0.0
		new(rate)
	end
end

function call(d::Dropout, X::Tensor)
	Knet.dropout(X, d.rate)
end

"""
Batch normalization layer (Ioffe and Szegedy, 2014). Normalizes the input at each batch,
i.e. applies a transformation that maintains the mean activation close to 0
and the activation standard deviation close to 1.

Finally, if activation is not nothing, it is applied to the outputs as well.
"""
struct BatchNorm <: Layer

  	moments
	activation

	function BatchNorm(activation=identity)
		new(Knet.bnmoments(), activation)
	end
end

function call(bn::BatchNorm, X::Tensor)
	bn.activation.(Knet.batchnorm(X, bn.moments))
end

"""
Beginning of allowing for a single model instance to run on multiple devices (model distribution)
This is still highly experimental and only tested on simple inference cases.
"""
struct ContextSwitch <: Layer

  	device::Symbol
	deviceId::Int
	dtype::Type

	function ContextSwitch(;device=getcontext().device,
							deviceId=getcontext().deviceId,
							dtype=getcontext().dtype)
		new(device,deviceId,dtype)
	end
end

function call(c::ContextSwitch, X::Tensor)
	setcontext(device=c.device, deviceId=c.deviceId, dtype=c.dtype)

	if c.device == :cpu
		X = convert(Array{c.dtype},X)
	elseif c.device == :gpu
		X = convert(Knet.KnetArray{c.dtype},X)
	end
	X
end



@debug "Loaded Core modules"
