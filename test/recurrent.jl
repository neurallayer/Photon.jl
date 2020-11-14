module RecurrentTests

using Photon, Test


gethistoric(steps=100, features=10) = KorA(randn(features,steps,4))


function basic_layer(T)

    d = gethistoric(100)

    l = T(20)
    pred = l(d)
    @test size(pred) == (20,4)

    l = T(20,2)
    pred = l(d)
    @test size(pred) == (20,4)

    l = T(20;dropout=0.5, bidirectional=true)
    pred = l(d)
    @test size(pred) == (20,4)

    l = T(20;dropout=0.5, last_only=false)
    pred = l(d)
    @test size(pred) == (20,100,4)

    l = T(20; initw=zeros, initb=zeros, initf=zeros)
    pred = l(d)
    @test size(pred) == (20,4)
    @test sum(pred) == 0.0

end


function train_lstm_model()

    model = Sequential(
        LSTM(50, last_only=false),
        Dense(100, relu),
        Dense(10)
    )

    w = Workout(model, MSELoss())

    X = [randn(10,20,8) for i in 1:10]
    Y = [randn(10,8)    for i in 1:10]
    data = zip(X,Y)
    for _ in 1:100
        train!(w, data;cb=SilentMeter())
        # GC.gc() # this line fixed some issue
    end
end



function lstm_model()

    model = Sequential(
        LSTM(20),
        Dense(100, relu),
        Dense(10)
    )

    pred = model(gethistoric(100))
    @test size(pred) == (10,4)

    pred = model(gethistoric(200))
    @test size(pred) == (10,4)

end

function lstm_model2()

    model = Sequential(
        LSTM(20, last_only=false),
        Dense(100, relu),
        Dense(10)
    )

    pred = model(gethistoric(100,10))
    @test size(pred) == (10,4)

    # @test_throws DimensionMismatch model(gethistoric(200,20))
end

function gru_model()

    model = Sequential(
        GRU(20,2;dropout=0.5),
        Dense(100, relu),
        Dense(10)
    )

    pred = model(gethistoric(100))
    @test size(pred) == (10,4)
end


function rnn_model()

    model = Sequential(
        RNN(20,2;activation=:relu, dropout=0.5),
        Dense(100, relu),
        Dense(10)
    )

    pred = model(gethistoric(100))
    @test size(pred) == (10,4)
end


@testset "Recurrent" begin
    train_lstm_model()
    basic_layer(LSTM)
    basic_layer(GRU)
    basic_layer(RNN)
    lstm_model()
    lstm_model2()
    gru_model()
    rnn_model()
end

end
