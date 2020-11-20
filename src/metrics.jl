module Metrics

using Photon: Tensor
using Statistics
import Knet

export Metric, BinaryAccuracy, OneHotBinaryAccuracy


abstract type Metric end


"""
Binary accuracy calculation
"""
struct BinaryAccuracy
    threshold

    BinaryAccuracy(;threshold=0.5) = new(threshold)
end

function (a::BinaryAccuracy)(y_pred::Tensor, y_true::Tensor)
    y_pred = y_pred .> a.threshold
    return mean(y_true .== y_pred)
end

"""
Binary accuracy for a onehot classification.
"""
struct OneHotBinaryAccuracy end

function (a::OneHotBinaryAccuracy)(y_pred::Tensor, y_true::Tensor)
    y_pred, y_true = Knet.mat(y_pred), Knet.mat(y_true)

    y_pred = mapslices(argmax,Knet.value(Array(y_pred)),dims=1)
    y_true = Array(y_true)

    mean(y_pred .== y_true)
end

end