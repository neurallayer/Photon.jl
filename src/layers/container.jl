
using MacroTools: @forward

"""
Common behavior for stacked layers that enables to access them as arrays
"""
abstract type StackedLayer <: Layer end

@forward StackedLayer.layers Base.getindex, Base.length, Base.first, Base.last,
  Base.iterate, Base.lastindex, Base.push!

add(model::StackedLayer, layer...) = push!(model::StackedLayer, layer...)



"""
Sequential layer allows to chain together a number of other layers.

# Usage

```julia
model = Sequential(Conv2D(100),MaxPool(),Dense(10))
```

"""
struct Sequential <:StackedLayer
    layers::Vector

	Sequential(blocks...) = new([blocks...])
end

function call(model::Sequential, X::Tensor)
	for layer in model.layers
		X = layer(X)
	end
	return X
end


"""
Concurrrent layer allows for stacking a number of other layers in parallel and
combining their results before returning it.

This layer will stack on the second last dimension.
So with 2D and 3D convolution this will be the channel layer (WxHxCxN). As a result
other dimensions have to the same.
"""
struct Concurrent <:StackedLayer
    layers::Vector

	Concurrent(blocks...) = new([blocks...])
end

function call(model::Concurrent, X::Tensor)
	out = []
	for layer in model.layers
		push!(out, layer(X))
	end
	cat(out..., dims=ndims(out[1])-1)
end


"""
Residual Layer works like a Sequential layer, however before returning the result
it will be combined with the orginal input (residual). This is a popular techique
in modern neural networds since it allows for better backpropagation.

This will stack on the second last dimension.
So with 2D and 3D convolution this will be the channel layer (WxHxCxN)
"""
struct Residual <:StackedLayer
    layers::Vector

	Residual(blocks...) = new([blocks...])
end

function call(model::Residual, X::Tensor)
	res = X
	for layer in model.layers
		X = layer(X)
	end
	cat(res, X, dims=ndims(res[1])-1)
end

@debug "Loaded Container modules"
