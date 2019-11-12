module Zoo

using Photon

include("alexnet.jl")
export AlexNet

include("densenet.jl")
export DenseNet121, DenseNet161, DenseNet169, DenseNet201

include("vgg.jl")
export VGG11, VGG13, VGG16, VGG19


end
