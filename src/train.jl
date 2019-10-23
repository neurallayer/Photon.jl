using Photon
using Knet
using Knet:minimize

using IterTools

include(Knet.dir("data", "mnist.jl"))

# model = VGG16(classes=10)

function get_model()
    Sequential(
        Conv2D(16, 3, activation=relu),
        Conv2D(64, 3, activation=relu),
        MaxPool2D(),
        Dense(32, activation=relu),
        Dense(10)
    )
end

count = 0
lv = 0.0
function loss(x, y)
    global count, lv
    count += 1
    l = nll(model(x), y)
    lv = 0.9*lv + 0.1*l
    if (count % 500) == 0
        println("Loss: ", lv)
    end
    return l
end


ctx.devType = :cpu
model = get_model()
dtrn, dtst = mnistdata(xtype=Array)
adam!(loss, ncycle(dtrn, 1))

ctx.devType = :gpu
model = get_model()
dtrn, dtst = mnistdata(xtype=KnetArray)

opt = Adam()

for step in minimize(loss,ncycle(dtrn, 2),opt)
    println(step)
    break
end


# adam!(loss, ncycle(dtrn, 10))
