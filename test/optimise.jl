module OptimTests

using Photon, Test
using Photon.Layers
using Photon.Optimisers: Adam, SGD, Momentum
using Photon.Callbacks: SilentMeter
using Photon.Losses

function get_model()
    Sequential(
        Conv2D(16, (3,3), :relu),
        Conv2D(32, (5,5), :relu),
        MaxPool2D(),
        Dense(64, :relu),
        Dense(10)
    )
end

function testOptim(opt)
    model = get_model()

    X = [randn(50,50,3,4) for _=1:5]
    Y = [randn(10,4) for _=1:5]

    workout = Workout(model, MSELoss(), opt)
    train!(workout, zip(X,Y), epochs=2, cb=SilentMeter())
    @test hasmetric(workout, :loss)
end

@testset "Optim" begin
    testOptim(Adam())
    testOptim(SGD())
    testOptim(Momentum())
end

end
