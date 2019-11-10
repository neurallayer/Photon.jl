"""
A dataset that contains random generated samples.
Ideal for quick testing of a model. Optionally you
can pass a sleep value to simulate IO blocking.

Examples:

	ds = TestDataset((28,28,1),(10,),1000)
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

	function ImageDataset(filenames, labels)
		try
			@eval import Images
		catch
			@warn "Package Images not installed"
		end
		new(filenames, labels)
	end

end


Base.length(ds::ImageDataset) = length(ds.filenames)

function Base.getindex(ds::ImageDataset, idx)
	filename = ds.filenames[idx]
	img = Images.load("docs/src/assets/juno_printscreen.png");
	img = Images.channelview(img);
	img = permutedims(img, [2,3,1]);
	img = convert(Array{Float32}, img);
	return (img, ds.labels[idx])
end
