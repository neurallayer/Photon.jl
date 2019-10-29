module TrainingTests

using Photon, Knet, Test

function simple_conv_model()
    model = Sequential(
        Conv2D(16, 3, activation=relu),
        Conv2D(32, 3, activation=relu),
        MaxPool2D(),
        Dense(100, activation=relu),
        Dense(10)
    )
    return model
end



function getdata(s=28)
    [(
        KorA(randn(Float32,s,s,1,16)),
        KorA(randn(Float32,10,16))
    ) for i=1:10]
end

function test_train()
    model = simple_conv_model()
    workout = Workout(model, mse, ADAM())

    data = getdata()
    fit!(workout, data, epochs=100)

    @test workout.epochs == 100
    @test workout.steps == (100 * length(data))
end

include("../src/models/densenet.jl")

function test_densenet(epochs, batches, device)
    ctx.devType = device
    model = DenseNet121()
    workout = Workout(model, mse, ADAM())

    minibatch = (KorA(randn(Float32,224,224,3,2)), KorA(randn(Float32,1000,2)))

    function randomdata()
        Channel() do channel
            for i in 1:batches
                put!(channel, minibatch)
            end
        end
    end

    fit!(workout, randomdata, epochs=epochs)
end

@testset "Training" begin
    test_train()
    if gpu() >= 0
        test_densenet(10,10,:gpu)
    end
    test_densenet(1,1,:cpu)
end

end
