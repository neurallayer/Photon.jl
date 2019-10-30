
module ConvTests

using Photon, Test
using Knet:relu
import Knet

getimages(s=224) = KorA(randn(ctx.dtype,s,s,3,4))



function simple_conv_model()
    model = Sequential(
        Conv2D(16, (3,3), relu),
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

@testset "Conv2D" begin
    simple_conv_model()
    adaptive_model()
end

end
