using Photon
using Knet

using IterTools

include(Knet.dir("data", "mnist.jl"))

# model = VGG16(classes=10)

function get_model()
    Sequential(
        Conv2D(16, (3,3), activation=relu),
        Conv2D(32, (5,5), activation=relu),
        Flatten(),
        Dense(100, activation=relu),
        Dense(10)
    )
end

count = 0

function loss(x, y)
    global count
    count += 1
    l = nll(model(x), y)
    if (count % 1000) == 0
        println("Loss: ", l)
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
adam!(loss, ncycle(dtrn, 1))
