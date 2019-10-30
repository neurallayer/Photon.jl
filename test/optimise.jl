module OptimTests

using Photon, Test
using Knet:relu



function get_model()
    Sequential(
        Conv2D(16, (3,3), relu),
        Conv2D(32, (5,5), relu),
        MaxPool2D(),
        Dense(100, relu),
        Dense(10)
    )
end

function testOptim(opt)
    model = get_model()
    X = [randn(ctx.dtype,50,50,3,4) for _=1:10]
    Y = [randn(ctx.dtype,10,4) for _=1:10]

    workout = Workout(model, mse, opt)
    fit!(workout, zip(X,Y), epochs=2)
    @test hasmetric(workout, :loss)
end

@testset "Optim" begin
    testOptim(ADAM())
    testOptim(SGD())
    testOptim(Momentum())
end

end
