module TrainingTests

using Photon, Test


function simple_conv_model()
    model = Sequential(
        Conv2D(16, 3, relu),
        Conv2D(32, 3, relu),
        MaxPool2D(),
        Dense(32, relu),
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
    workout = Workout(model, mse)

    data = getdata()
    fit!(workout, data, epochs=2)

    @test workout.epochs == 2
    @test workout.steps == (2 * length(data))
    @test hasmetric(workout, :loss)
end

function test_channel()
    model = simple_conv_model()
    workout = Workout(model, mse)

    minibatch = (KorA(randn(Float32,30,30,3,4)), KorA(randn(Float32,10,4)))

    function randomdata()
        Channel() do channel
            for i in 1:4
                put!(channel, minibatch)
            end
        end
    end

    fit!(workout, randomdata, epochs=2)
    @test workout.epochs == 2
    @test hasmetric(workout, :loss)
end

@testset "Training" begin
    resetContext()
    test_train()
    test_channel()
end

end
