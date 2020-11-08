## Convolutional layers

expand(N, i::Tuple) = i
expand(N, i::Integer) = ntuple(_ -> i, N)

"""
Convolutional layer that serves as the base for Conv2D and Conv3D
"""
mutable struct Conv <: LazyLayer
    channels::Int
    kernel_size::Union{Int,Tuple}
    activation
    padding::Union{Int,Tuple}
    strides::Union{Int,Tuple}
    dilation
    built::Bool
    use_bias::Bool
	init::NamedTuple
    params::NamedTuple
end

function Conv(
    channels::Int,
    kernel_size::Union{Int,Tuple},
    activation = identity;
    padding = 0,
    strides = 1,
    dilation = 1,
    use_bias = true,
	initw = Knet.xavier,
	initb = zeros
)
    @assert channels > 0 "Conv layer should have more then 0 channels"
    Conv(
        channels,
        kernel_size,
        activation,
        padding,
        strides,
        dilation,
        false,
        use_bias,
		(w=initw, b=initb),
        (w=nothing, b=nothing)
    )
end

function build(layer::Conv, shape::Shape, atype)
    input_channels = shape[end]
    rank = length(shape) - 1
    kernel_size = expand(rank, layer.kernel_size)
    w = getparam(atype, kernel_size..., input_channels, layer.channels, init=layer.init.w)
    b = nothing
    if layer.use_bias
        b = getparam(atype, repeat([1],rank)..., layer.channels, 1, init = layer.init.b)
    end
    layer.params = (w=w,b=b)
end

function call(c::Conv, x::Tensor)
    w,b = c.params

    x = Knet.conv4(
        w,
        x,
        padding = c.padding,
        stride = c.strides,
        dilation = c.dilation,
    )
    if c.use_bias
        c.activation.(x .+ b)
    else
        c.activation.(x)
    end
end


"""
Utility function to determine the output size of a convolutional
layer given a certain input size and configuration of a convolutional layer.
This function works even if the weights of the layer are not initialized.

For example:
	c = Conv2D(16, (3,3), strides=3, padding=1)
	output_size(c, (224,224)) # output is (75, 75)

"""
function output_size(c::Conv, input_size)
    result = []
    l = length(input_size)
    padding = expand(l, c.padding)
    strides = expand(l, c.strides)
    dilation = expand(l, c.dilation)
    kernel_size = expand(l, c.kernel_size)

    for (i, elem) in enumerate(input_size)
        dimension = floor(
            Int,
            (elem + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) /
            strides[i],
        ) + 1
        push!(result, dimension)
    end
    tuple(result...)
end


"""
1D convolution layer (e.g. spatial convolution over a timeseries).

This layer creates a convolution kernel that is convolved with the layer
input to produce a tensor of outputs. If use_bias is true, a bias vector
is created and added to the outputs. Finally, if activation is not nothing,
it is applied to the outputs as well.
"""
Conv1D = Conv



"""
2D convolution layer (e.g. spatial convolution over images).

This layer creates a convolution kernel that is convolved with the layer
input to produce a tensor of outputs. If use_bias is true, a bias vector
is created and added to the outputs. Finally, if activation is not nothing,
it is applied to the outputs as well.
"""
Conv2D = Conv


"""
3D convolution layer (e.g. spatial convolution over volumes).

This layer creates a convolution kernel that is convolved with the layer input
to produce a tensor of outputs. If use_bias is true, a bias vector is created
and added to the outputs. Finally, if activation is not nothing, it is
applied to the outputs as well.
"""
Conv3D = Conv

"""
ConvTranspose layer
"""
mutable struct ConvTranspose <: LazyLayer
    channels::Int
    kernel_size::Union{Int,Tuple}
    activation
    padding::Union{Int,Tuple}
    strides::Union{Int,Tuple}
    dilation
    built::Bool
    use_bias::Bool
	init::NamedTuple
    params::NamedTuple
end

function ConvTranspose(
    channels::Int,
    kernel_size::Union{Int,Tuple},
    activation = identity;
    padding = 0,
    strides = 1,
    dilation = 1,
    use_bias = true,
	initw = Knet.xavier,
	initb = zeros
)
    @assert channels > 0
    ConvTranspose(
        channels,
        kernel_size,
        activation,
        padding,
        strides,
        dilation,
        false,
        use_bias,
		(w=initw, b=initb),
        (w=nothing, b=nothing)
    )
end

function build(layer::ConvTranspose, shape::Tuple, atype)
    input_channels = shape[end]
    rank = length(shape) - 1
    kernel_size = expand(rank, layer.kernel_size)
    w = getparam(atype, kernel_size..., layer.channels, input_channels,init=layer.init.w)
    b = nothing
    if layer.use_bias
        b = getparam(atype, repeat([1],rank)..., layer.channels, 1, init=layer.init.b)
    end
    layer.params = (w=w,b=b)
end

function call(c::ConvTranspose, x::Tensor)
    w,b = c.params

    x = Knet.deconv4(
        w,
        x,
        padding = c.padding,
        stride = c.strides,
        dilation = c.dilation,
    )
    if c.use_bias
        c.activation.(x .+ b)
    else
        c.activation.(x)
    end
end

"""
Transposed 2D convolution layer (sometimes called Deconvolution).

The need for transposed convolutions generally arises from the desire to use a
transformation going in the opposite direction of a normal convolution,
i.e., from something that has the shape of the output of some convolution
to something that has the shape of its input while maintaining a connectivity
pattern that is compatible with said convolution.
"""
Conv2DTranspose = ConvTranspose

"""
Transposed 3D convolution layer (sometimes called Deconvolution).

The need for transposed convolutions generally arises from the desire to use a
transformation going in the opposite direction of a normal convolution,
i.e., from something that has the shape of the output of some convolution
to something that has the shape of its input while maintaining a connectivity
pattern that is compatible with said convolution.
"""
Conv3DTranspose = ConvTranspose

## AvgPool layer
"""
Pooling layers
"""
abstract type PoolingLayer <: Layer end

struct AvgPool <: PoolingLayer
    pool_size::Union{Int,Tuple}
    padding::Union{Int,Tuple}
    strides::Union{Int,Tuple}
end

function AvgPool(pool_size=2; padding = 0, strides = pool_size)
    AvgPool(pool_size, padding, strides)
end

function call(p::AvgPool, X)
    Knet.pool(
        X,
        window = p.pool_size,
        padding = p.padding,
        stride = p.strides,
        mode = 0,
    )
end

AvgPool1D = AvgPool

"""Average pooling operation for two dimensional (spatial) data."""
AvgPool2D = AvgPool

"""Average pooling operation for three dimensional (spatial) data."""
AvgPool3D = AvgPool

"""
Adaptive Average Pool has a fixed size output and enables creating a convolutional
network that can be used for multiple image formats.
"""
struct AdaptiveAvgPool <: PoolingLayer
    output_size::Tuple
end

function call(m::AdaptiveAvgPool, x::Tensor)
    s = []
    for idx = 1:length(m.output_size)
        push!(s, size(x, idx) - m.output_size[idx] + 1)
    end
    s = tuple(s...)
    return Knet.pool(x, window = s, padding = 0, stride = 1, mode = 0)
end


## MaxPool layer

struct MaxPool <: PoolingLayer
    pool_size::Union{Int,Tuple}
    padding::Union{Int,Tuple}
    strides::Union{Int,Tuple}
    nanOpt::Int
end

function MaxPool(pool_size=2; padding = 0, strides = pool_size, nanOpt = 0)
    MaxPool(pool_size, padding, strides, nanOpt)
end

function call(p::MaxPool, X::Tensor)
    Knet.pool(
        X,
        window = p.pool_size,
        padding = p.padding,
        maxpoolingNanOpt = p.nanOpt,
        stride = p.strides,
        mode = 1,
    )
end

MaxPool1D = MaxPool

"""Max pooling operation for two dimensional (spatial) data."""
MaxPool2D = MaxPool

"""Max pooling operation for three dimensional (spatial) data."""
MaxPool3D = MaxPool

"""
Adaptive MaxPool has a fixed size output and enables creating a convolutional
network that can be used for different image sizes.
"""
struct AdaptiveMaxPool <: PoolingLayer
    output_size::Tuple
end

function call(m::AdaptiveMaxPool, x::Tensor)
    s = []
    for idx = 1:length(m.output_size)
        push!(s, size(x, idx) - m.output_size[idx] + 1)
    end
    s = tuple(s...)
    return Knet.pool(x, window = s, padding = 0, stride = 1, mode = 1)
end

@debug "Loaded Convolutional modules"
