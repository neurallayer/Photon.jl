module DenseTests

using Photon, Test
using Photon.Layers
import Knet
import Photon.Layers: Dense, Sequential
import Photon: hasgpu, KorA

function dense_model_cpu()

    function run_model(data)
        model = Sequential(Dense(10), Dense(1))
        model(data)
    end

    data = randn(10, 16)
    pred = run_model(data)
    @test size(pred) == (1, 16)
    @test typeof(pred) == Array{Float64,2}

    data = randn(Float32, 10, 16)
    pred = run_model(data)
    @test size(pred) == (1, 16)
    @test typeof(pred) == Array{Float32,2}

end


function dense_model_gpu()
    function run_model(data)
        model = Sequential(Dense(10), Dense(1))
        model(data)
    end

    data = KorA(randn(10, 16))
    pred = run_model(data)
    @test size(pred) == (1, 16)
    @test typeof(pred) == Knet.KnetArray{Float32,2}
end


@testset "Dense" begin
    dense_model_cpu()
    hasgpu() && dense_model_gpu()
end


end
