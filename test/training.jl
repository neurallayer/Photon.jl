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


function test_save_load()
    m = simple_conv_model()
    X = KorA(randn(Float32,28,28,1,4))
    pred = m(X)
    workout = Workout(m, MSELoss())
    f = saveWorkout(workout)
    workout2 = loadWorkout(f)
    rm(f, force=true)
    pred2 = workout2.model(X)
    @assert pred == pred2
end


function getdata(s=28)
    [(
        KorA(randn(Float32,s,s,1,16)),
        KorA(randn(Float32,10,16))
    ) for i=1:10]
end

function test_train()
    model = simple_conv_model()
    workout = Workout(model, MSELoss())

    data = getdata()
    fit!(workout, data, epochs=2)

    @test workout.epochs == 2
    @test workout.steps == (2 * length(data))
    @test hasmetric(workout, :loss)
end

function test_train_valid()
    model = simple_conv_model()
    workout = Workout(model, MSELoss())

    data_tr = getdata()
    data_val = getdata()
    fit!(workout, data_tr, data_val, epochs=2)

    @test hasmetric(workout, :loss)
    @test hasmetric(workout, :val_loss)
end


function test_channel()
    model = simple_conv_model()
    workout = Workout(model, MSELoss())

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
    test_train_valid()
    test_channel()
    test_save_load()
end

end
