
# Should find more elgant method to find required atype
_get_basetype(x::Type{Knet.KnetArray{T,N}}) where {T,N} = Knet.KnetArray{T}
_get_basetype(x::Type{Knet.AutoGrad.Result{Knet.KnetArray{T,N}}}) where {T,N} = Knet.KnetArray{T}
_get_basetype(x::Type{Array{T,N}}) where {T,N} = Array{T}
_get_basetype(x::Type{Knet.AutoGrad.Result{Array{T,N}}}) where {T,N} = Array{T}

function getparam(atype::Type, d...;init=Knet.xavier)
	atype = _get_basetype(atype)
	Knet.Param(atype(init(d...)))
end


const RegisteredActivations = IdDict{String, Any}(
	"relu" => Knet.relu,
	"elu" => Knet.elu,
	"selu" => Knet.selu,
	"sigm" => Knet.sigm,
	"sigmoid" => Knet.sigm,
	"identity" => identity
)

get_activation(f) = f
function get_activation(f::Union{String, Symbol})
	f = string(f)
	if ! haskey(RegisteredActivations, f)
		@warn "unkown activtion function, using relu instead" fn=f
		return Knet.relu
	else
		return RegisteredActivations[f]
	end
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

"""
Lazy layers will be created during the first invocation. This means that a development time 
you don't need to specify much, since most aspects will be figured out based on the past data.
"""
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
        build(layer, size(X)[1:end-1], typeof(X))
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
		activation = get_activation(activation)
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

function build(layer::Dense, shape::Shape, atype)
	# nInput = length(shape) > 1 ? *(shape...) : shape[1]
	nInput = *(shape...)

    w = getparam(atype, layer.units, nInput, init=layer.init.w)
	b = nothing
    if layer.use_bias
		b = getparam(atype, layer.units, init=layer.init.b)
	end
	layer.params = (w=w,b=b)
end


"""
Flattening Layer. Photon by default already has flattening functionality
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


"""
Reshape the shape of a tensor. 
"""
mutable struct Reshape <: Layer
	dims
	Reshape(dims::Tuple) = new(dims)
end

function call(layer::Reshape, X::Tensor)
	dims = collect(layer.dims)
    batchsize = size(X)[end]
    newdim = push!(dims, batchsize)
	reshape(X, Tuple(newdim))
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
		activation = get_activation(activation)
		new(Knet.bnmoments(), activation)
	end
end

function call(bn::BatchNorm, X::Tensor)
	bn.activation.(Knet.batchnorm(X, bn.moments))
end

@debug "Loaded Core modules"
