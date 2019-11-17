module MetricsTests

using Photon, Test


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

function test_core()
    model = simple_conv_model()
    workout = Workout(model, MSELoss())

    data = getdata()
    fit!(workout, data, epochs=2)
    h = history(workout, :loss)

    @test h isa Tuple
    @test length(h[1]) == length(h[2])
end

function test_metrics()
    y_pred = rand(10,16)
    y_true = rand(0:1,10,16)
    b = BinaryAccuracy()
    acc = b(y_pred, y_true)

    o = OneHotBinaryAccuracy()
    cc = o(y_pred, y_true)

end


@testset "Metrics" begin
    resetContext()
    test_core()
    test_metrics()
end

end
