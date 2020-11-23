module Optimisers

    # For now we just use the Knet optimisers and export them for uniform API
    export Adam, SGD, Momentum, Nesterov, Adagrad, Rmsprop, Adadelta

    using Knet: Adam, SGD, Momentum, Nesterov, Adagrad, Rmsprop, Adadelta
    
end