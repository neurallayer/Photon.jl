def binary_accuracy(y_true, y_pred, threshold=0.5):
    if threshold != 0.5:
        threshold = K.cast(threshold, y_pred.dtype)
        y_pred = K.cast(y_pred > threshold, y_pred.dtype)
    return K.mean(K.equal(y_true, K.round(y_pred)), axis=-1)

using Statistics


struct BinaryAccuracy
    name::String
    threshold

    BinaryAccuracy(name="acc", threshold=0.5) = new(name,threshold)
end

function (a::BinaryAccuracy)(y_pred, y_true)
    y_pred = y_pred .> a.threshold
    return mean(y_true .== y_pred)
end


a = KorA(rand(Float64,10,10))
b = KorA(float(rand(0:1,10,10)))


ba = BinaryAccuracy()
z = ba(a,b)

Knet.value(z)
