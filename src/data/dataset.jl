"""
Test dataset that contains random generated samples.
Ideal for testing.

Examples:

	ds = TestDataset((28,28,1),(10,),10000)
"""
struct TestDataset <: Dataset
    x
    y

	function TestDataset(shapeX::Tuple, shapeY::Tuple, samples::Int)
		dtype = getContext().dtype
		x = [randn(dtype, shapeX...) for _ in 1:samples]
		y = [randn(dtype, shapeY...) for _ in 1:samples]
		new(x,y)
	end
end

Base.length(ds::TestDataset) = length(ds.x)
Base.getindex(ds::TestDataset, idx) = return (ds.x[idx], ds.y[idx])


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
Dataset that loads the input from file.
"""
struct FileDataset{A,B} <: Dataset
    filenames::A
    labels::B
end

Base.length(ds::FileDataset) = length(ds.filenames)
function Base.getindex(ds::FileDataset, idx)
	filename = ds.filenames[idx]
	new(nothing, nothing)
end
