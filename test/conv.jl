
module ConvTests

using Photon, Test

using Photon, Test
import Knet: relu
import Photon.Layers: Dense, Sequential, Conv1D, Conv2D, Conv3D, Flatten, Conv2DTranspose, MaxPool2D, AvgPool2D, MaxPool2D, AdaptiveAvgPool, AdaptiveMaxPool, output_size
import Photon: hasgpu, KorA


getimages(s=224) = KorA(randn(s,s,3,4))


function simple_1D_model()
    model = Sequential(
        Conv1D(16, 3, relu),
        Conv1D(32, 5, relu),
        Dense(100, relu),
        Dense(10)
    )

    data = KorA(randn(100,3,4))
    pred = model(data)
    @test size(pred) == (10,4)

end



function simple_1D_model2()
    model = Sequential(
        Conv1D(16, 3, relu),
        Conv1D(32, 5, relu),
        Dense(100, relu),
        Dense(10)
    )

    data = KorA(randn(100,3,4))
    pred = model(data)
    @test size(pred) == (10,4)

end



function simple_3D_model()
    model = Sequential(
        Conv3D(16, 3, relu),
        Conv3D(32, 5, relu),
        Dense(100, relu),
        Dense(10)
    )

    data = KorA(randn(20,20,20,3,4))
    pred = model(data)
    @test size(pred) == (10,4)

end

function test_base_Conv2D()
    X = getimages()

    Y = Conv2D(16, (3,3))(X)
    @test Y !== nothing

    Y = Conv2D(16, (3,3), relu)(X)
    @test Y !== nothing

    Y = Conv2D(16, 3; padding=1, strides=2, dilation=2)(X)
    @test Y !== nothing

end

function convtranspose_model()
    model = Sequential(
        Conv2D(16, (3,3), relu; padding=1, strides=2, dilation=2),
        Conv2D(32, (5,5), relu),
        Conv2DTranspose(32, (5,5)),
        Conv2DTranspose(16,3; padding=1, strides=2, dilation=1),
        Dense(50, relu),
        Dense(10)
    )
    pred = model(getimages())
    @test size(pred) == (10,4)
end


function simple_conv2D_model()
    model = Sequential(
        Conv2D(16, (3,3), relu; padding=1, strides=2, dilation=2),
        Conv2D(32, (5,5), relu),
        Flatten(),
        Dense(100, relu),
        Dense(10)
    )
    pred = model(getimages())
    @test size(pred) == (10,4)
end


function get_output_sizeTest()
    layer = Conv2D(16, (3,3), padding=1, strides=2, dilation=2)
    s = output_size(layer,(224,224))
    @test length(s) == 2
end


function adaptive_max_model()
    # Adaptive model
    model = Sequential(
        Conv2D(16, (3,3), relu),
        MaxPool2D(),
        Conv2D(32, (5,5), relu),
        AvgPool2D(),
        AdaptiveMaxPool((10,10)),
        Flatten(),
        Dense(100, relu),
        Dense(10)
    )

    pred = model(getimages(224))
    @test size(pred) == (10,4)

    pred = model(getimages(100))
    @test size(pred) == (10,4)

end


function adaptive_avg_model()
    # Adaptive model
    model = Sequential(
        Conv2D(16, (3,3), relu),
        MaxPool2D(),
        Conv2D(32, (5,5), relu),
        AvgPool2D(),
        AdaptiveAvgPool((10,10)),
        Flatten(),
        Dense(100, relu),
        Dense(10)
    )

    pred = model(getimages(224))
    @test size(pred) == (10,4)

    pred = model(getimages(100))
    @test size(pred) == (10,4)

end

@testset "Conv" begin
    # simple_1D_model()
    test_base_Conv2D()
    simple_conv2D_model()
    convtranspose_model()
    adaptive_avg_model()
    adaptive_max_model()
    get_output_sizeTest()
    simple_3D_model()
end


end
