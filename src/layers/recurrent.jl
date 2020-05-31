

"""
Quick thin wrapper around the Knet RNN implementation. Ideally would like to bring
it more inline with other layers, so Photon creates the weights and use Knet
as a stateless function.

The Knet API:

```julia
RNN(inputSize, hiddenSize;
             h=nothing, c=nothing,
             handle=gethandle(),
             numLayers=1,
             dropout=0.0,
             skipInput=false,     # CUDNN_LINEAR_INPUT = 0, CUDNN_SKIP_INPUT = 1
             bidirectional=false, # CUDNN_UNIDIRECTIONAL = 0, CUDNN_BIDIRECTIONAL = 1
             rnnType=:lstm,       # CUDNN_RNN_RELU = 0, CUDNN_RNN_TANH = 1, CUDNN_LSTM = 2, CUDNN_GRU = 3
             dataType=Float32,    # CUDNN_DATA_FLOAT  = 0, CUDNN_DATA_DOUBLE = 1, CUDNN_DATA_HALF   = 2
             algo=0,              # CUDNN_RNN_ALGO_STANDARD = 0, CUDNN_RNN_ALGO_PERSIST_STATIC = 1, CUDNN_RNN_ALGO_PERSIST_DYNAMIC = 2
             seed=0,              # seed=0 for random init, positive integer for replicability
             winit=xavier,
             binit=zeros,
             finit=ones,        # forget bias for lstm
             usegpu=(gpu()>=0),
             )
```
"""


mutable struct Recurrent <: LazyLayer
    hidden_size::Int
    num_layers::Int
    ops
    built::Bool
    last_only::Bool
    init::NamedTuple
    mode
end


function build(l::Recurrent, shape::Tuple)
    inputSize = shape[1]

    l.ops = Knet.RNN(
        inputSize,
        l.hidden_size,
        numLayers = l.num_layers,
        dataType = getcontext().dtype,
        usegpu = is_on_gpu(),
        rnnType = l.mode,
        winit = l.init.w,
        binit = l.init.b,
        finit = l.init.f,
    )
end

function call(layer::Recurrent, X::Tensor)
    output = layer.ops(X)
    if layer.last_only
        output[:, end, :]
    else
        output
    end
end

"""
Applies a multi-layer Elman RNN with :tanh or :relu non-linearity to an
input sequence.

For each element in the input sequence, each layer computes the following
function:

``h_t = \\text{tanh}(W_{ih} x_t + b_{ih} + W_{hh} h_{(t-1)} + b_{hh})``

"""
function RNN(
    hidden_size::Int,
    num_layers = 1;
    activation=:tanh,
    dropout = 0.0,
    bidirectional = false,
    last_only = true,
    initw = Knet.xavier,
    initb = zeros,
    initf = ones
)
    @assert activation in [:tanh, :relu] "Only :tanh and :relu are supported"
    Recurrent(hidden_size, num_layers, undef, false, last_only,
    (w=initw, b=initb, f=initf), activation)
end


@doc raw"""
Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

For each element in the input sequence, each layer computes the following
function:


``\begin{array}{ll}
i_t = sigmoid(W_{ii} x_t + b_{ii} + W_{hi} h_{(t-1)} + b_{hi})
\\f_t = sigmoid(W_{if} x_t + b_{if} + W_{hf} h_{(t-1)} + b_{hf})
\\g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hc} h_{(t-1)} + b_{hg})
\\o_t = sigmoid(W_{io} x_t + b_{io} + W_{ho} h_{(t-1)} + b_{ho})
\\c_t = f_t * c_{(t-1)} + i_t * g_t
\\h_t = o_t * \tanh(c_t)
\end{array}``

# Usage

```julia
layer = LSTM(50)
layer = LSTM(50, 2, bidirectional=true)
```
"""
function LSTM(
    hidden_size::Int,
    num_layers = 1;
    dropout = 0.0,
    bidirectional = false,
    last_only = true,
    initw = Knet.xavier,
    initb = zeros,
    initf = ones
)
    Recurrent(hidden_size, num_layers, undef, false, last_only,
    (w=initw, b=initb, f=initf), :lstm)
end


@doc raw"""
Applies a multi-layer gated recurrent unit (GRU) RNN to an input sequence.

For each element in the input sequence, each layer computes the following
function:

``\begin{array}{ll}
r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{(t-1)} + b_{hr})
\\z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{(t-1)} + b_{hz})
\\n_t = \tanh(W_{in} x_t + b_{in} + r_t * (W_{hn} h_{(t-1)}+ b_{hn}))
\\h_t = (1 - z_t) * n_t + z_t * h_{(t-1)}
\end{array}``

# Usage

```julia
layer = GRU(50)
layer = GRU(50, 2, bidirectional=true)
```

"""
function GRU(
    hidden_size::Int,
    num_layers = 1;
    dropout = 0.0,
    bidirectional = false,
    last_only = true,
    initw = Knet.xavier,
    initb = zeros,
    initf = ones
)
    Recurrent(hidden_size, num_layers, undef, false, last_only,
    (w=initw, b=initb, f=initf), :gru)
end


@debug "Loaded Recurrent modules"
