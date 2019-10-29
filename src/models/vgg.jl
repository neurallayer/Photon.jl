
using Knet:relu

function _make_features(layers, filters, batch_norm)
    featurizer = Sequential()
    for (filter, num) in zip(filters, layers)
        for _ = 1:num
            add(featurizer, Conv2D(filter, (3, 3), padding = 1))
            if batch_norm
                add(featurizer, BatchNorm())
            end
            add(featurizer, Activation(relu))
        end
        add(featurizer, MaxPool2D())
    end
    return featurizer
end


struct VGG <: Layer
    features
    output

    function VGG(layers, filters; classes = 1000, batch_norm = false)
        features = _make_features(layers, filters, batch_norm)
        add(features, Flatten())
        add(features, Dense(4096, activation = relu))
        add(features, Dropout(0.5))
        add(features, Dense(4096, activation = relu))
        add(features, Dropout(0.5))
        output = Dense(classes)
        new(features, output)
    end
end

function (model::VGG)(x)
    x = model.features(x)
    model.output(x)
end


vgg_spec = Dict(
    11 => ([1, 1, 2, 2, 2], [64, 128, 256, 512, 512]),
    13 => ([2, 2, 2, 2, 2], [64, 128, 256, 512, 512]),
    16 => ([2, 2, 3, 3, 3], [64, 128, 256, 512, 512]),
    19 => ([2, 2, 4, 4, 4], [64, 128, 256, 512, 512]),
)

VGG11(; classes = 1000, batch_norm = true) =
    VGG(vgg_spec[11]..., classes = classes, batch_norm = batch_norm)
VGG13(; classes = 1000, batch_norm = true) =
    VGG(vgg_spec[13]..., classes = classes, batch_norm = batch_norm)
VGG16(; classes = 1000, batch_norm = true) =
    VGG(vgg_spec[16]..., classes = classes, batch_norm = batch_norm)
VGG19(; classes = 1000, batch_norm = true) =
    VGG(vgg_spec[19]..., classes = classes, batch_norm = batch_norm)


@info "Loaded VGG models"
