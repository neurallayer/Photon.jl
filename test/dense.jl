module DenseTests

using Photon, Test
import Knet


function dense_model_cpu()

    function run_model(data)
        model = Sequential(Dense(10), Dense(1))
        model(data)
    end

    setcontext(device = :cpu, dtype = Float32)
    ctx = getcontext()
    data = randn(ctx.dtype, 10, 16)
    pred = run_model(data)
    @test size(pred) == (1, 16)
    @test typeof(pred) == Array{Float32,2}

    setcontext(device = :cpu, dtype = Float64)
    data = randn(ctx.dtype, 10, 16)
    pred = run_model(data)
    @test size(pred) == (1, 16)
    @test typeof(pred) == Array{Float64,2}

end


function dense_model_gpu()
    function run_model(data)
        model = Sequential(Dense(10), Dense(1))
        model(data)
    end

    setcontext(device = :gpu, dtype = Float32)
    ctx = getcontext()
    data = KorA(randn(ctx.dtype, 10, 16))
    pred = run_model(data)
    @test size(pred) == (1, 16)
    @test typeof(pred) == Knet.KnetArray{Float32,2}

end


function splitted_dense_model()
    function run_model(data)
        model = Sequential(Dense(10), ContextSwitch(device = :cpu), Dense(1))
        model(data)
    end

    setcontext(device = :gpu)
    ctx = getcontext()
    data = KorA(randn(ctx.dtype, 10, 16))
    pred = run_model(data)
    @test size(pred) == (1, 16)
    @test typeof(pred) == Array{Float32,2}

end


@testset "Dense" begin
    resetcontext()
    dense_model_cpu()
    if hasgpu()
        dense_model_gpu()
        splitted_dense_model()
    end
end


end
