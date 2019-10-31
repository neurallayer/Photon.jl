module LossTests

using Photon, Test

function test_distance(loss)
    p = randn(10,10)
    y = randn(10,10)
    r = loss(p,y)
    @test r > 0

    r = loss(p,p)
    @test r == 0

    p = randn(10,1,10)
    y = randn(1,10,10)
    r = loss(p,y)
    @test r > 0

    p = KorA(randn(10,10))
    y = KorA(randn(10,10))
    r = loss(p,y)
    @test r > 0

    r = loss(p,p)
    @test r == 0

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
    test_distance(MAELoss())
    test_distance(MSELoss())
    test_classification(CrossEntropyLoss())
    test_classification(BCELoss())
end



end
