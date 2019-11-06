
using Photon


struct OneHotBinaryAccuracy
    name::String

    OneHotBinaryAccuracy(name="acc") = new(name)
end

function (a::BinaryAccuracy2)(y_pred, y_true)
    y_pred, y_true = mat(y_pred), mat(y_true)
    if a.argmax
        y_pred = mapslices(argmax,Array(arr),dims=1)
        y_true = Knet.argmaxarray(y_true)
    else
        y_pred = y_pred .> a.threshold
    end
    return mean(y_true .== y_pred)
end


a = KorA(rand(Float64,10,10))
b = KorA(float(rand(0:1,10,10)))


ba = BinaryAccuracy()
z = ba(a,b)

using Statistics


Knet.value(z)

a = KorA(rand(Float32,10,16))


b = KorA(rand(0:1,10,16))

a .== b

argmax(a,dims=1)[4][2]

function amax(arr)
    mapslices(argmax,Array(arr),dims=1)
end

mean(mean(amax(a) .== Array(b), dims=1))

import Knet

Knet.argmaxarray(a)

@doc Knet.argmaxarray
