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


function test_core()
    model = simple_conv_model()
    workout = Workout(model, MSELoss())

    data = getdata()
    train!(workout, data, epochs=2)
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

function test_algo()

    pred   = [[0.3 0.7]; [0. 1.]; [0.4 0.6]]'
    labels = [[1 0];[0 1];[0 1]]'
    loss = CrossEntropyLoss()
    @assert loss(pred, labels) ≈ 0.5715992760

    pred   = reshape(Array([3, -0.5, 2, 7]), (4,1))
    labels = reshape(Array([2.5, 0.0, 2, 8]), (4,1))
    loss = L1Loss()
    @assert loss(pred,labels) ≈ 0.5

    pred   = reshape(Array([3, -0.5, 2, 7]), (4,1))
    labels = reshape(Array([2.5, 0.0, 2, 8]), (4,1))
    loss = L2Loss()
    @assert loss(pred,labels) ≈ 0.375

    pred   = [[0.3 0.7]; [0. 1.]; [0.4 0.6]]
    labels = [[1 0];[0 1];[0 1]]
    loss = BinaryAccuracy()
    @assert loss(pred, labels) ≈ 0.666666666666

end

@testset "Metrics" begin
    resetcontext()
    test_callbacks()
    test_core()
    test_metrics()
    test_algo()
end

end
