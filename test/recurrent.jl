module RecurrentTests

using Photon, Test
using Knet:relu
import Knet



gethistoric(steps=100, features=10) = KorA(randn(ctx.dtype,features,steps,4))


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
        RNN(20,2;activation=:relu, dropout=0.5),
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

end
