module CallbackTests

using Photon, Test
using Photon.Layers
using Photon.Callbacks
using Photon.Metrics
using Knet: relu
using Photon.Losses


function simple_conv_model()
    model = Sequential(
        Conv2D(4, 3, relu),
        Conv2D(8, 3, relu),
        MaxPool2D(),
        Dense(16, relu),
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


function test_callbacks()
    model = simple_conv_model()
    workout = Workout(model, MSELoss())

    data = getdata()
    filename = "myworkout.dat"
    rm(filename, force=true)

    train!(workout, data, epochs=2, cb=AutoSave(:loss, filename))
    @assert isfile(filename)
    rm(filename, force=true)

    train!(workout, data, epochs=2, cb=EpochSave(filename))
    @assert isfile(filename)
    rm(filename, force=true)

    val_data = getdata()
    workout = Workout(model, MSELoss())
    train!(workout, data, val_data, epochs=10, cb=EarlyStop(:val_loss))
    @assert workout.epochs < 10
end



@testset "Callbacks" begin
    test_callbacks()
end

end
