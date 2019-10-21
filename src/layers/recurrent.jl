"""
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
"""


mutable struct Recurrent <: LazyLayer
    hidden_size::Int
    num_layers::Int
    ops
    initialized::Bool
    last_only::Bool
    mode
end

function LSTM(
    hidden_size,
    num_layers = 1;
    dropout = 0.0,
    bidirectional = false,
    last_only = true,
)
    Recurrent(hidden_size, num_layers, undef, false, last_only, :lstm)
end

function GRU(
    hidden_size,
    num_layers = 1;
    dropout = 0.0,
    bidirectional = false,
    last_only = true,
)
    Recurrent(hidden_size, num_layers, undef, false, last_only, :gru)
end

function RNN_TANH(
    hidden_size,
    num_layers = 1;
    dropout = 0.0,
    bidirectional = false,
    last_only = true,
)
    Recurrent(hidden_size, num_layers, undef, false, last_only, :tanh)
end

function RNN_RELU(
    hidden_size,
    num_layers = 1;
    dropout = 0.0,
    bidirectional = false,
    last_only = true,
)
    Recurrent(hidden_size, num_layers, undef, false, last_only, :relu)
end


function initlayer(l::Recurrent, X)
    inputSize = size(X, 1)
    usegpu = ctx.devType == :gpu
    l.ops = Knet.RNN(
        inputSize,
        l.hidden_size,
        numLayers = l.num_layers,
        dataType = ctx.dataType,
        usegpu = usegpu,
        rnnType = l.mode,
    )
end

function forward(layer::Recurrent, X)
    output = layer.ops(X)
    if layer.last_only
        output[:, end, :]
    else
        output
    end
end

@info "Loaded Recurrent modules"
