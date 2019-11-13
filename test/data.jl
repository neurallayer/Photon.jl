
module DataTests

using Photon, Test, JLD2

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


function test_jld()
	rm("example.jld2", force=true)

	jldopen("example.jld2", "w") do file
	    file["image4"] = (randn(Float32,28,28,1), rand(0:9,1))
	    file["image5"] = (randn(Float32,28,28,1), rand(0:9,1))
	end

	f = jldopen("example.jld2", "r")
	ds = JLDDataset(f)
	(X,Y) = ds[1]
	close(f)
	@assert size(X) == (28,28,1)
	@assert size(Y) == (1,)
	rm("example.jld2", force=true)
end

function test_output()

	samples = 9
	X1 = [zeros(Float32,28,28,1) for _ = 1:samples]
	Y1 = [ones(2) for _ = 1:samples]
	ds = ArrayDataset(X1,Y1)

	dataloader = Dataloader(ds,4)
	@assert dataloader.batchsize == 4
	@assert length(dataloader) == samples ÷ 4

	for (X,Y) in dataloader
		@assert size(X,4) == dataloader.batchsize
		@assert size(Y,2) == dataloader.batchsize
		@assert sum(X) == 0.0
		@assert sum(Y) == 1.0 * 2 * dataloader.batchsize
	end

end

function test_dataloader()
    ds = TestDataset((28,28,1), (10,), 50)
	dl = Dataloader(ds)
	sample = first(dl)
	@test  size(sample[1]) == (28,28,1,8)
	@test  size(sample[2]) == (10,8)
end


function test_training()
    ds = TestDataset((28,28,1), (10,), 100)
	dl = Dataloader(ds)

	model = Sequential(
        Conv2D(16, 3, relu),
        Conv2D(32, 3, relu),
        MaxPool2D(),
        Dense(32, relu),
        Dense(10)
    )

	workout = Workout(model, MSELoss())
	fit!(workout, dl, epochs=2)
	@assert workout.steps == (100÷8)*2

end


function test_threading(sleep)
    ds = TestDataset((28,28,1), (10,), 100; sleep=sleep)
	dl = Dataloader(ds, 32)

	model = Sequential(
        Conv2D(16, 3, relu),
        Conv2D(32, 3, relu),
        MaxPool2D(),
        Dense(32, relu),
        Dense(10)
    )

	workout = Workout(model, MSELoss())
	@time fit!(workout, dl, epochs=2)
	@assert workout.steps == (100÷32)*2
end


function test_transformers()
	ds = TestDataset((10,),(1,),100)
	ds = ds |> NoisingTransfomer()
	sample = ds[1]
	@assert size(sample[1]) == (10,)

	ds = TestDataset((100,100,3),(20,20,1),100)
	ds = ds |> ImageCrop((50,50),(10,10))
	sample = ds[10]
	@assert size(sample[1]) == (50,50,3)
	@assert size(sample[2]) == (10,10,1)

	X = [randn(100,100,3) for _ in 1:10]
	Y = [rand(0:9) for _ in 1:10]

	ds = ArrayDataset(X,Y)
	ds = ds |> OneHotEncoder(0:9)
	sample = ds[5]
	@assert size(sample[2]) == (10,)
end

@testset "Data" begin
    test_dataset()
	test_jld()
	test_output()
	test_dataloader()
	test_training()
	test_threading(0.01)
	test_threading(0.02)
	test_transformers()
end


end
