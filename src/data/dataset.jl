"""
A dataset that contains random generated samples.
Ideal for quick testing of a model.

Examples:

	ds = TestDataset((28,28,1),(10,),10000)
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
Base dataset that contains two arrays.
"""
struct ArrayDataset{A,B} <: Dataset
    x::A
    y::B
end

Base.length(ds::ArrayDataset) = length(ds.x)
Base.getindex(ds::ArrayDataset, idx) = return (ds.x[idx], ds.y[idx])


"""
Dataset that loads an image from a file.
"""
struct ImageDataset{A,B} <: Dataset
    filenames::A
    labels::B
end

Base.length(ds::ImageDataset) = length(ds.filenames)

function Base.getindex(ds::ImageDataset, idx)
	# TODO
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
