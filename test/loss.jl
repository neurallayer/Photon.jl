module LossTests

using Photon, Test

function test_distance(loss)
    p = randn(10,8)
    y = randn(10,8)
    r = loss(p,y)
    @test r > 0
    @test loss(p,y) == loss(y,p)

    r = loss(p,p)
    @test r == 0

    p = randn(10,1,8)
    y = randn(1,10,8)
    r = loss(p,y)
    @test r > 0

    p = KorA(randn(10,8))
    y = KorA(randn(10,8))
    r = loss(p,y)
    @test loss(p,y) == loss(y,p)
    @test r > 0

    r = loss(p,p)
    @test r == 0

end

function test_nondistance(loss)
    p = randn(10,8)
    y = randn(10,8)
    r = loss(p,y)
    @test r isa Number

    p = randn(10,1,8)
    y = randn(1,10,8)
    r = loss(p,y)
    @test r isa Number

    p = KorA(randn(10,8))
    y = KorA(randn(10,8))
    r = loss(p,y)
    @test r isa Number

    @test loss(p,p) <= loss(p,y)

end


function test_classification(loss)
    p = Float32.(rand(0:0.0001:1.0,10,16))
    y = Float32.(rand(0:1, 10,16))
    r = loss(p, y)
    @test r > 0

    r = loss(KorA(p), KorA((y)))
    @test r > 0

end

@testset "Loss" begin
    test_distance(L1Loss())
    test_distance(L1Loss(reduction=sum))
    test_distance(L2Loss())
    test_distance(LNLoss(3))

    test_nondistance(PseudoHuberLoss())

    test_classification(HingeLoss())
    test_classification(CrossEntropyLoss())
    test_classification(BCELoss())
end



end
