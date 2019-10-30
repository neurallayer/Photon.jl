
using Knet:relu

mutable struct AlexNet <: Layer
    features
    output
end


function AlexNet(; classes = 1000)
    features = Sequential(
        Conv2D(64, (11, 11), relu, strides = 4, padding = 2),
        MaxPool2D(pool_size=3, strides = 2),
        Conv2D(192, (5, 5), relu, padding = 2),
        MaxPool2D(pool_size=3, strides = 2),
        Conv2D(384, (3, 3), relu, padding = 1),
        Conv2D(256, (3, 3), relu, padding = 1),
        Conv2D(256, (3, 3), relu, padding = 1),
        MaxPool2D(pool_size=3, strides = 2),
        Flatten(),
        Dense(4096, relu),
        Dropout(0.5),
        Dense(4096, relu),
        Dropout(0.5),
    )

    output = Dense(classes)
    AlexNet(features, output)
end

function (m::AlexNet)(X)
    X = m.features(X)
    m.output(X)
end


@info "Loaded AlexNet model"
