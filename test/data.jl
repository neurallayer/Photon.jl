
module DataTests

using Photon, Test

resetContext()
ctx = getContext()

getimages(s=224) = KorA(randn(ctx.dtype,s,s,3,4))


function test_dataset()
    ds = TestDataset((28,28,1), (10,), 50)
	@test length(ds) == 50
	sample = ds[1]
	@test  size(sample[1]) == (28,28,1)
	@test  size(sample[2]) == (10,)
end


function test_output()

	samples = 9
	X1 = [zeros(Float32,28,28,1) for _ = 1:samples]
	Y1 = [ones(2) for _ = 1:samples]
	ds = ArrayDataset(X1,Y1)

	dataloader = Dataloader(ds,4)
	@assert dataloader.batchsize == 4
	
	for (X,Y) in dataloader
		@assert size(X,4) == dataloader.batchsize
		@assert size(Y,2) == dataloader.batchsize
		@assert sum(X) == 0.0
		@assert sum(Y) == 1.0 * 2 * dataloader.batchsize
	end

end

function test_dataloader()
    ds = TestDataset((28,28,1), (10,), 50)
	dl = Dataloader(ds,8)
	sample = first(dl)
	@test  size(sample[1]) == (28,28,1,8)
	@test  size(sample[2]) == (10,8)
end

@testset "Data" begin
    test_dataset()
	test_output()
	test_dataloader()
end


end
