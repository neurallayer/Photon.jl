
using MacroTools: @forward

"""
Common behavior for stacked layers that enables to access them as arrays
"""
abstract type StackedLayer <: Layer end

@forward StackedLayer.layers Base.getindex, Base.length, Base.first, Base.last,
  Base.iterate, Base.lastindex, Base.push!

add(model::StackedLayer, layer...) = push!(model::StackedLayer, layer...)



"""
Sequential
"""
struct Sequential <:StackedLayer
    layers

	function Sequential(blocks...)
		new([blocks...])
	end
end

function forward(model::Sequential,X)
	for layer in model.layers
		X = layer(X)
	end
	return X
end


"""
Concurrrent
"""
struct Concurrent <:StackedLayer
    layers

	function Concurrent(blocks...)
		new([blocks...])
	end
end

function forward(model::Concurrent, X)
	out = []
	for layer in model.layers
		push!(out, layer(X))
	end
	cat(out..., dims=ndims(out[1])-1)
end


"""
Residual Layer. This will stack on the second last dimension. So with
and 2D convolution this will be the channel layer (WxHxCxN)
"""
struct Residual <:StackedLayer
    layers

	function Residual(blocks...)
		new([blocks...])
	end
end

function forward(model::Residual, X)
	res = X
	for layer in model.layers
		X = layer(X)
	end
	cat(res, X, dims=ndims(res[1])-1)
end

@info "Loaded Container modules"
