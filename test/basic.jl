module BasicTests

using Photon, Test

function KorATest()
    a = randn(10,10)
    b = KorA(a)
    @test size(a) == size(b)
    @test isapprox(sum(a), sum(b))
end

function utils()
    a = randn(32,10)
    b = batchlast(a)
    @test  a[1,:] == b[:,1]

    a = randn(10, 32)
    b = batchfirst(a)
    @test  a[:,1] == b[1,:]

    a = randn(32, 10)
    @test batchfirst(batchlast(a)) == a
end

@testset "Basic" begin
    KorATest()
    utils()
end

end
