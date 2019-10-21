


mutable struct Context
	devType::Symbol
	devId::Int
	dataType::Type
end

global ctx = Context(:gpu,0,Float32)




function is_on_gpu()
	ctx.devType == :gpu
end


function getparam(d...;init=xavier)
	et = ctx.dataType
	atype = ctx.devType == :gpu ? KnetArray{et} : Array{et}
	Param(atype(init(d...)))
end


abstract type Layer end
function (layer::Layer)(X)
	@debug "Calling $(typeof(layer))" size(X)
    forward(layer, X)
end

"""
LazyLayer will initialize the first time it sees input data. It will typically
infer its required parameters based on the size, type and context of this input data.

A LazyLayer needs to implement forward and initlayer functions and need an attribute
called initialized.
"""
abstract type LazyLayer <: Layer end
function (layer::LazyLayer)(X)
    if ! layer.initialized
		@debug "Initializing $(typeof(layer))" size(X) layer
        initlayer(layer, X)
		layer.initialized = true
    end
	@debug "Calling $(typeof(layer))" size(X)
    forward(layer, X)
end



"""
Fully connected layer with an optinal bias
"""
mutable struct Dense <:LazyLayer
    w
    b
    units
    activation
    initialized::Bool
	use_bias::Bool
end

function Dense(units; activation=identity, use_bias=true, flatten=true)
	@assert units > 0 "Dense layer should have more then 0 units"
    Dense(undef, undef, units, activation, false, use_bias)
end

function forward(layer::Dense, X)
	if layer.use_bias
    	layer.activation.(layer.w*X .+ layer.b)
	else
		layer.activation.(layer.w*X)
	end
end

function initlayer(layer::Dense, X)
	# t = eltype(X)
    layer.w = getparam(layer.units, size(X,1))
    if layer.use_bias
		layer.b = getparam(layer.units, init=zeros)
	else
		layer.b  = nothing
	end
end



"""
Flattening Layer
"""
mutable struct Flatten <: Layer
	dims
	Flatten(dims=nothing) = new(dims)
end

function forward(layer::Flatten, X)
	layer.dims == nothing ? mat(X) : mat(X, dims=layer.dims)
end



## Activation
mutable struct Activation <:Layer
	activation
end

function forward(c::Activation, x)
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

function forward(d::Dropout,X)
	dropout(X, d.rate)
end

## Batch Normalization Layer
struct BatchNorm <: Layer

  	moments

	function BatchNorm()
		new(bnmoments())
	end
end

function forward(bn::BatchNorm, X)
	batchnorm(X, bn.moments)
end







@info "Loaded Core modules"
