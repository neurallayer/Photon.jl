
using Random
struct Dataset{A,B}
    x::A
    y::B
end

Base.length(ds::Dataset) = length(ds.x)
Base.getindex(ds::Dataset, idx) = return (ds.x[idx], ds.y[idx,:])


struct Dataloader
    dataset
    batchsize::Int
    shuffle::Bool
    datatypes::Tuple
    datasizes::Tuple
    dataaxes::Tuple

    function Dataloader(dataset, batchsize=16, shuffle=true)
        sample = dataset[1]
        datatypes = typeof.(sample)
        datasizes = size.(sample)
        dataaxes = axes.(sample)
        new(dataset, batchsize, shuffle, datatypes, datasizes, dataaxes)
    end
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

        minibatch = [Vector{datatype}(undef, bs) for datatype in dl.datatypes]

        Threads.@threads for i in 1:bs
            idx = i + count - 1
            sample = dl.dataset[idx]
            for idx in 1:length(sample)
                minibatch[idx][i] = sample[idx]
            end
        end

        return ((minibatch), (idxs, count + bs))
end

samples = 1000
X = [randn(Float32,28,28,1) for i = 1:samples]
Y = [UInt8(2) for i = 1:samples]

ds = Dataset(X,Y)
dataloader = Dataloader(ds)
for batch in dataloader
end

first(dataloader)
