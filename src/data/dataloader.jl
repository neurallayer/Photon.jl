
#=
function update_mb!(arr::AbstractArray, elem::AbstractArray, idx::Int)
	@assert size(arr)[1:end-1] == size(elem) "$(size(arr)) $(size(elem))"
	idxs = Base.OneTo.(size(elem))
	arr[idxs..., idx] = elem
end


function update_mb!(t::Tuple, elems::Tuple, idx::Int)
	@assert length(t) == length(elems)
	for (arr,elem) in zip(t,elems)
		update_mb!(arr, elem, idx)
	end
end

# Create a minibatch that has a similar size to the structure returned
# from the dataset.
create_mb(arr::AbstractArray, batchsize) = similar(arr, size(arr)..., batchsize)
create_mb(t::Tuple, batchsize)= Tuple(collect(create_mb(elem, batchsize) for elem in t))
=#

"""
A dataloader can be passed to the fit! function as data provider. It will call
a dataset to get a single sample and combines them into a minibatch. The dataloader
itself is an iterator.

It will do this using threading, so when a dataset will read some samples from disk
this won't become a botttleneck.

Example:

	ds = ImageDataset(filenames,labels)
	dl = Dataloader(ds, 16, shuffle=false)
	fit!(workout, dl)

"""
struct Dataloader
    dataset::Dataset
    batchsize::Int
    shuffle::Bool

    Dataloader(dataset::Dataset, batchsize=8; shuffle=true) =
        new(dataset, batchsize, shuffle)
end

Base.length(dl::Dataloader) = length(dl.dataset) ÷ dl.batchsize

function Base.iterate(dl::Dataloader, state=undef)
    maxl = length(dl.dataset)
    bs = dl.batchsize

    if state == undef
        idxs = dl.shuffle ? Random.shuffle(1:maxl) : 1:max1
        state = (idxs,1)
    end
    idxs, count = state

    if count > (maxl-bs) return nothing end

	l = Threads.SpinLock()
	minibatch = nothing

    Threads.@threads for i in 1:bs

		idx = i + count - 1
		sample = dl.dataset[idx]
		@assert sample isa Tuple "Datasets should return Tuples, not $(typeof(sample))"

		if minibatch == nothing
			Threads.lock(l)
			if minibatch == nothing
				minibatch = create_mb(sample, bs)
			end
			Threads.unlock(l)
		end

		update_mb!(minibatch, sample, i)
    end
	Threads.unlock(l)
    return ((minibatch), (idxs, count + bs))
end
