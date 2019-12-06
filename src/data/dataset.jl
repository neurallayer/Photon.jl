
using ..Photon: getContext

"""
Datasets can load and transform data and can be used in a pipeline.
Typically they would have length and index methods implemented and need to return
only a single sample. The MiniBatch transformer would collect these single samples
and put them into a mini batch.

```julia
data = SomeDataset() |> SomeTransformer() |> MiniBatch()
train!(workout, data)
```

Please note that in case all your data would fit in memory, you can feed it
directly to the train! function (so a dataset is not required)

```julia
X = [randn(10,10,16) for i in 1:100]
Y = [randn(1,16) for i in 1:100]
train!(workout, zip(X,Y))
```

"""
abstract type Dataset end


"""
A dataset that contains random generated samples. Ideal for quick testing
of a model. The random values will be drawn from a normal distribution.

You need to provide the shape of X, Y and the number of samples that the
dataset should contain. Optionally you can specify a sleep value to simulate
IO blocking.

# Usage

```julia
xshape = (28,28,1)
yshape = (10,)
ds = TestDataset(xshape, yshape, 60000)

ds = TestDataset((100,), (1,), 1000, sleep=0.1)
```
"""
struct TestDataset <: Dataset
    x::Vector{Array}
    y::Vector{Array}
	sleep::Float64

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
Dataset that contains two vectors, one for the input data and one for the labels.
"""
struct VectorDataset{A,B} <: Dataset
    x::A
    y::B
end

Base.length(ds::VectorDataset) = length(ds.x)
Base.getindex(ds::VectorDataset, idx) = return (ds.x[idx], ds.y[idx])



"""
Dataset that loads data from a JLD(2) file. The format
that is stored is expected to be in the shape of: (X,Y)

Note: Tested with JLD2 library only.

Example:
========

```julia
using JLD2

jldopen("example.jld2", "w") do file
    file["image1"] = (randn(Float32,28,28,1), rand(0:9,1))
    file["image2"] = (randn(Float32,28,28,1), rand(0:9,1))
end

f = jldopen("example.jld2", "r")
ds = JLD2Dataset(f)
ds[1]
```
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
Dataset that loads data from a JuliaDB. Not yet implemented.
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
Dataset that loads an single image from a file and optionally resizes the image.
The labels are passed as is.

# Usage

```julia
ds = ImageDataset(filenames, labels, resize=(200,200))
```
"""
struct ImageDataset <: Dataset
    filenames::Vector{String}
    labels::Vector
	resize::Union{Nothing,Tuple}

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
	# img = Images.load(filename) has multi-thread issue
    img = ImageMagick.load(filename)
	img = Images.channelview(img)
	img = permutedims(img, [2,3,1])
	img = convert(Array{Float32}, img)
	if ds.resize !== nothing
		img = Images.imresize(img, ds.resize...)
	end
	return (img, ds.labels[idx])
end



"""
Dataset that retrieves is data from a dataframe. The provided column names
for X and Y can be either a single Symbol or a Vector of Symbols.

# Usage

```julia
df = DataFrame(randn(4,20))
ds = DFDataset(df, :x1, :x2)
ds = DFDataset(df, [:x1, :x3], [:x2,:x4])
```
"""
struct DFDataset <: Dataset
	df
    X::Vector{Symbol}
	Y::Vector{Symbol}

	DFDataset(df,X,Y) = new(df, makeArray(X), makeArray(Y))
end

Base.length(ds::DFDataset) = size(df,1)

function Base.getindex(ds::DFDataset, idx)
	df = ds.df
	(Vector(df[idx, ds.X]), Vector(df[idx, ds.Y]))
end
