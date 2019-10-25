module PhotonTest

using Photon
using Knet


## Simple conv model
using Test
@info "Running Unit tests"

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
        KnetArray(randn(Float32,s,s,1,16)),
        KnetArray(randn(Float32,10,16))
    ) for i=1:10]
end


function test_train()

    ctx.devType = :gpu
    model = simple_conv_model()
    workout = Workout(model, mse, ADAM())

    data = getdata()
    fit!(workout, data, 100)

    @test workout.epochs == 100
    @test workout.steps == (100 * length(data))
end

@testset "Training" begin
    test_train()
end

end
