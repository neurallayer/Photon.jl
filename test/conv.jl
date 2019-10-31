
module ConvTests

using Photon, Test
using Knet:relu
import Knet

getimages(s=224) = KorA(randn(ctx.dtype,s,s,3,4))


function simple_1D_model()
    model = Sequential(
        Conv1D(16, 3, relu),
        Conv1D(32, 5, relu),
        Dense(100, relu),
        Dense(10)
    )

    data = KorA(randn(ctx.dtype,100,3,4))
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

    data = KorA(randn(ctx.dtype,20,20,20,3,4))
    pred = model(data)
    @test size(pred) == (10,4)

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


function simple_conv_model()
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

function adaptive_model()
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
    simple_conv_model()
    # convtranspose_model()
    adaptive_model()
    simple_3D_model()
end

end
