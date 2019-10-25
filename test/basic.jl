module BasicTests

using Photon, Knet, Test

function getimages(s=224)
    images = randn(ctx.dataType,s,s,3,4)
    images = ctx.devType == :gpu ? KnetArray(images) : images
end

function gethistoric(steps=100, features=10)
    images = randn(ctx.dataType,features,steps,4)
    images = ctx.devType == :gpu ? KnetArray(images) : images
end


function simple_conv_model()
    model = Sequential(
        Conv2D(16, (3,3), activation=relu),
        Conv2D(32, (5,5), activation=relu),
        Flatten(),
        Dense(100, activation=relu),
        Dense(10)
    )
    pred = model(getimages())
    @test size(pred) == (10,4)
end

function adaptive_model()
    # Adaptive model
    model = Sequential(
        Conv2D(16, (3,3), activation=relu),
        MaxPool2D(),
        Conv2D(32, (5,5), activation=relu),
        AvgPool2D(),
        AdaptiveAvgPool((10,10)),
        Flatten(),
        Dense(100, activation=relu),
        Dense(10)
    )

    pred = model(getimages(224))
    @test size(pred) == (10,4)

    pred = model(getimages(100))
    @test size(pred) == (10,4)

end

@testset "Conv2D" begin
    simple_conv_model()
    adaptive_model()
end


function lstm_model()
    # Adaptive model
    model = Sequential(
        LSTM(20),
        Dense(100, activation=relu),
        Dense(10)
    )

    pred = model(gethistoric(100))
    @test size(pred) == (10,4)

    pred = model(gethistoric(200))
    @test size(pred) == (10,4)

end

function lstm_model2()
    # Adaptive model
    model = Sequential(
        LSTM(20, last_only=false),
        Dense(100, activation=relu),
        Dense(10)
    )

    pred = model(gethistoric(100))
    @test size(pred) == (10,4)

    @test_throws DimensionMismatch model(gethistoric(200))
end

function gru_model()
    # Adaptive model
    model = Sequential(
        GRU(20,2;dropout=0.5),
        Dense(100, activation=relu),
        Dense(10)
    )

    pred = model(gethistoric(100))
    @test size(pred) == (10,4)
end


function rnn_model()
    # Adaptive model
    model = Sequential(
        RNN_RELU(20,2;dropout=0.5),
        Dense(100, activation=relu),
        Dense(10)
    )

    pred = model(gethistoric(100))
    @test size(pred) == (10,4)
end


@testset "Recurrent" begin
    lstm_model()
    lstm_model2()
    gru_model()
    rnn_model()
end


function dense_model()
    function run_model(data)
        model = Sequential(
            Dense(10),
            Dense(1)
        )
        model(data)
    end

    ctx.dataType = Float32
    ctx.devType = :cpu
    data = randn(ctx.dataType, 10,16)
    pred = run_model(data)
    @test size(pred) == (1,16)
    @test typeof(pred) == Array{Float32,2}

    ctx.dataType = Float64
    ctx.devType = :cpu
    data = randn(ctx.dataType, 10,16)
    pred = run_model(data)
    @test size(pred) == (1,16)
    @test typeof(pred) == Array{Float64,2}

    ctx.dataType = Float32
    ctx.devType = :gpu
    data = randn(Float32, 10,16)
    data = gpu() >= 0 ? KnetArray(data) : data
    pred = run_model(data)
    @test size(pred) == (1,16)
    @test typeof(pred) == KnetArray{Float32,2}

end


function splitted_dense_model()
    function run_model(data)
        model = Sequential(
            Dense(10),
            ContextSwitch(devType=:cpu),
            Dense(1)
        )
        model(data)
    end

    ctx.devType = :gpu
    data = randn(Float32, 10,16)
    data = gpu() >= 0 ? KnetArray(data) : data
    pred = run_model(data)
    @test size(pred) == (1,16)
    @test typeof(pred) == Array{Float32,2}

end

@testset "Dense" begin
    dense_model()
    splitted_dense_model()
end


end
