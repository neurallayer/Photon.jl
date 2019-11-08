
using Random

abstract type Dataset end


function update_mb!(arr::Array, elem::Array, idx)
	@assert size(arr)[1:end-1] == size(elem)
	idxs = Base.OneTo.(size(elem))
	arr[idxs..., idx] = elem
end


function update_mb!(t::Tuple, elems::Tuple, idx)
	@assert length(t) == length(elems)
	for (arr,elem) in zip(t,elems)
		update_mb!(arr, elem, idx)
	end
end

create_mb(arr::Array, batchsize) = similar(arr, size(arr)..., batchsize)
create_mb(t::Tuple, batchsize)= Tuple(collect(create_mb(elem, batchsize) for elem in t))


struct Dataloader
    dataset::Dataset
    batchsize::Int
    shuffle::Bool

    Dataloader(dataset::Dataset, batchsize=16, shuffle=true) =
        new(dataset, batchsize, shuffle)
end

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

struct MyDataset{A,B} <: Dataset
    x::A
    y::B
end

Base.length(ds::MyDataset) = length(ds.x)
Base.getindex(ds::MyDataset, idx) = return (ds.x[idx], ds.y[idx])


samples = 1000
X = [zeros(Float32,28,28,1) for _ = 1:samples]
Y = [ones(2) for _ = 1:samples]
ds = MyDataset(X,Y)

dataloader = Dataloader(ds)
for batch in dataloader
	println(size.(batch))
	@assert sum(batch[1]) == 0.0
	@assert sum(batch[2]) == 1.0 * 2 * dataloader.batchsize
end

first(dataloader)
