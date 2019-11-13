module Data

include("dataset.jl")
export Dataset,ImageDataset, ArrayDataset, TestDataset, JLDDataset, JuliaDBDataset

include("transformer.jl")
export ImageCrop, NoisingTransfomer, onehot, OneHotEncoder

include("dataloader.jl")
export Dataloader

end
