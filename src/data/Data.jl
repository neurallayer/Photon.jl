module Data

using Random

include("dataset.jl")
export Dataset,ImageDataset, ArrayDataset, TestDataset, JLDDataset,
        JuliaDBDataset, DFDataset

include("transformer.jl")
export Transformer, ImageCrop, NoisingTransfomer, onehot, OneHotEncoder,
        MiniBatch, Normalizer, Subset, Split

end
