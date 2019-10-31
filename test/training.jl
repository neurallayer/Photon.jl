module TrainingTests

using Photon, Test
using Knet:relu

function simple_conv_model()
    model = Sequential(
        Conv2D(16, 3, relu),
        Conv2D(32, 3, relu),
        MaxPool2D(),
        Dense(50, relu),
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
    fit!(workout, data, epochs=10)

    @test workout.epochs == 10
    @test workout.steps == (10 * length(data))
    @test hasmetric(workout, :loss)
end

include("../src/models/densenet.jl")

function test_densenet(epochs, batches, device)
    setContext(device=device, dtype=Float32)
    model = DenseNet121(classes=100)
    workout = Workout(model, mse, ADAM())

    # use only 1 image per batch, otherwise Travis won't finish in time
    minibatch = (KorA(randn(Float32,224,224,3,1)), KorA(randn(Float32,100,1)))

    function randomdata()
        Channel() do channel
            for i in 1:batches
                put!(channel, minibatch)
            end
        end
    end

    fit!(workout, randomdata, epochs=epochs)
    @test workout.epochs == epochs
    @test hasmetric(workout, :loss)
end

@testset "Training" begin
    resetContext()
    test_train()
    if hasgpu()
        test_densenet(10,10,:gpu)
    end
    test_densenet(1,1,:cpu)
end

end
