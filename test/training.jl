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
    f = saveworkout(workout)
    workout2 = loadworkout(f)
    rm(f, force=true)
    pred2 = workout2.model(X)
    @assert pred == pred2
end


function test_predict()
    m = simple_conv_model()
    X = KorA(randn(Float32,28,28,1,4))
    workout = Workout(m, MSELoss())
    p1 = predict(workout, X, makebatch=false)
    p2 = predict(workout, X[:,:,:,1])

    # @assert p1[:,:,:,1] == p2
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
    train!(workout, data, epochs=2, cb=SilentMeter())

    @test workout.epochs == 2
    @test workout.steps == (2 * length(data))
    @test hasmetric(workout, :loss)
end

function test_train_valid()
    model = simple_conv_model()
    workout = Workout(model, MSELoss())

    data_tr = getdata()
    data_val = getdata()
    train!(workout, data_tr, data_val, epochs=2, cb=SilentMeter())

    @test hasmetric(workout, :loss)
    @test hasmetric(workout, :val_loss)
end


function test_validate()
    model = simple_conv_model()
    workout = Workout(model, MSELoss())
    e, s = workout.epochs, workout.steps

    data = getdata()
    validate(workout, first(data))

    @test workout.epochs == e
    @test workout.steps == s
    @test hasmetric(workout, :val_loss)
end


function test_gradients()
    model = simple_conv_model()
    workout = Workout(model, MSELoss())
    e, s = workout.epochs, workout.steps

    data = getdata()
    g = gradients(workout, first(data))

    @test workout.epochs == e
    @test workout.steps == s
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

    train!(workout, randomdata, epochs=2,cb=SilentMeter())
    @test workout.epochs == 2
    @test hasmetric(workout, :loss)
end

@testset "Training" begin
    resetcontext()
    test_train()
    test_train_valid()
    test_validate()
    test_predict()
    test_gradients()
    test_channel()
    test_save_load()
end

end
