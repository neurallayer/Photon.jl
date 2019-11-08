using Photon
import Knet

m = Sequential(
        Dense(20),
        Dense(10)
        )


X = KorA(randn(Float32,10,16))
pred = m(X)

ps = Knet.params(m)

# Lets not store gradients
Photon.zerograd!(ps)
ps[1]
# Move weights to CPU so the serialization contains all
tmp_ps = IdDict()
for p in ps
    tmp_ps[p] = p.value
    p.value = Array(p.value)
end

SGD

m.layers[1].params.w.value

m.layers[1].params.w.value = Array(m.layers[1].params.w.value)

ps[1].value

Array(ps[1].value)

ps[1].value = Array(ps[1].value)

ps

workout = Workout(m, MSELoss(), ADAM())

saveWorkout(workout)

workout2 = loadWorkout()

pred2 = workout2.model(X)

pred2 == pred


import Serialization
import Knet

struct MyParam
    value
end

function Serialization.serialize(s::Serialization.AbstractSerializer, t::Knet.Param)
    # println("Called")
    p = MyParam(Array(t.value))
    Serialization.serialize(s,p)
end


function Serialization.deserialize(s::Serialization.AbstractSerializer, t::MyParam)
    println("Called")
    p = MyParam(Array(t.value))
    Serialization.serialize(s,p)
end


m = Sequential(
        Dense(20),
        Dense(10)
        )


X = KorA(randn(Float32,10,16))
pred = m(X)

Serialization.serialize("test.ser", m);

n = Serialization.deserialize("test.ser")

n

typeof(typeof(m.layers[1].params.w))

n.layers[1].params.w


import Knet

mutable struct Param3{T} <: Knet.Value{T}
    value
    grad
end

Tensor = Union{Knet.KnetArray, Array}

p = Knet.Param{Tensor}(randn(10,10))


function test()
    t::Tensor = randn(10,10)
    Knet.Param(t)
end

test()
