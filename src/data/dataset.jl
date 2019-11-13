
using ..Photon: getContext

"""
Datasets are repsonsible for loading a single sample. They need to have length and
index methods implemented. Hence, they can all be passed to a Dataloader which can load
multiple samples parallelly using threading.

Pleae note that in case your whole dataset would fit in memory, you can feed it
directly to the fit! function (so a dataset is no requirement)

```julia
X = [randn(10,10,16) for i in 1:100]
Y = [randn(1,16) for i in 1:100]
fit!(workout, zip(X,Y))
```

Otherwise the combination of dataset/dataloader is your best bet.
"""
abstract type Dataset end


"""
A dataset that contains random generated samples.
Ideal for quick testing of a model.

You need to provide the shape of X, Y and the number of
samples that it should contain. Optionally you
can specify a sleep value to simulate IO blocking.

# Usage

```julia
ds = TestDataset((28,28,1),(10,),1000)
ds = TestDataset((100,),(1,),100, sleep=0.1)
```
"""
struct TestDataset <: Dataset
    x
    y
	sleep

	function TestDataset(shapeX::Tuple, shapeY::Tuple, samples::Int; sleep=0)
		dtype = getContext().dtype
		x = [randn(dtype, shapeX...) for _ in 1:samples]
		y = [randn(dtype, shapeY...) for _ in 1:samples]
		new(x,y,sleep)
	end
end

Base.length(ds::TestDataset) = length(ds.x)

function Base.getindex(ds::TestDataset, idx)
	ds.sleep > 0 && sleep(ds.sleep)
	return (ds.x[idx], ds.y[idx])
end

"""
Base dataset that contains two vectors.
"""
struct ArrayDataset{A,B} <: Dataset
    x::A
    y::B
end

Base.length(ds::ArrayDataset) = length(ds.x)
Base.getindex(ds::ArrayDataset, idx) = return (ds.x[idx], ds.y[idx])



"""
Dataset that loads data from a JLD(2) file. The format
that is stored is expected to be in the shape of: (X,Y)

Note: Tested with JLD2 library only.

Example:
========

	using JLD2

	jldopen("example.jld2", "w") do file
	    file["image1"] = (randn(Float32,28,28,1), rand(0:9,1))
	    file["image2"] = (randn(Float32,28,28,1), rand(0:9,1))
	end

	f = jldopen("example.jld2", "r")
	ds = JLD2Dataset(f)
	ds[1]

"""
struct JLDDataset <: Dataset
	f
    keys

	JLDDataset(f) = new(f,keys(f))
end

Base.length(ds::JLDDataset) = length(ds.keys)

function Base.getindex(ds::JLDDataset, idx)
	key = ds.keys[idx]
	ds.f[key]
end


"""
Dataset that loads data from a JuliaDB
"""
struct JuliaDBDataset{A,B} <: Dataset
    filenames::A
    labels::B
end

Base.length(ds::JuliaDBDataset) = length(ds.filenames)

function Base.getindex(ds::JuliaDBDataset, idx)
	# TODO
end


"""
Dataset that loads an image from a file.
"""
struct ImageDataset <: Dataset
    filenames
    labels
	resize

	function ImageDataset(filenames, labels; resize=nothing)
		@assert length(filenames) == length(labels)
		try
			@eval import Images
			@eval import ImageMagick
		catch
			@warn "Package Images or ImageMagick not installed"
		end
		new(filenames, labels, resize)
	end

end

Base.length(ds::ImageDataset) = length(ds.filenames)

function Base.getindex(ds::ImageDataset, idx)
	filename = ds.filenames[idx]
	# img = Images.load(filename) has mult-tread issues
    img = ImageMagick.load(filename)
	img = Images.channelview(img)
	img = permutedims(img, [2,3,1])
	img = convert(Array{Float32}, img)
	if ds.resize != nothing
		img = Images.imresize(img, ds.resize...)
	end
	return (img, ds.labels[idx])
end
