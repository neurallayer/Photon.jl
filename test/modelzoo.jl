module ZooTests

using Photon, Test

include("../src/models/vgg.jl")
include("../src/models/alexnet.jl")
include("../src/models/densenet.jl")


"""
############################################
#### Start of Testing of Specific Models ###
############################################
"""

getimages(s=224) = KorA(randn(ctx.dtype,s,s,3,4))

function test_vgg()
    model = VGG16()
    images = getimages()
    pred = model(images)
    @test size(pred) == (1000,size(images,4))
end

@testset "VGG" begin
    test_vgg()
end


function test_alexnet()
    model = AlexNet(classes=500)
    images = getimages()
    pred = model(images)
    @test size(pred) == (500,size(images,4))
end

@testset "AlexNet" begin
    test_alexnet()
end


function test_densenet()
    model = DenseNet169()
    images = getimages()
    pred = model(images)
    @test size(pred) == (1000, size(images,4))

    model = DenseNet121(classes=500)
    images = getimages()
    pred = model(images)
    @test size(pred) == (500, size(images,4))

end

@testset "DenseNet" begin
    test_densenet()
end

end
