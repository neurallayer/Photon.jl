mutable struct AlexNet <: Layer
    features
    output
end


function AlexNet(; classes = 1000)
    features = Sequential(
        Conv2D(64, (11, 11), strides = 4, padding = 2, activation = relu),
        MaxPool2D(pool_size = 3, strides = 2),
        Conv2D(192, (5, 5), padding = 2, activation = relu),
        MaxPool2D(pool_size = 3, strides = 2),
        Conv2D(384, (3, 3), padding = 1, activation = relu),
        Conv2D(256, (3, 3), padding = 1, activation = relu),
        Conv2D(256, (3, 3), padding = 1, activation = relu),
        MaxPool2D(pool_size = 3, strides = 2),
        Flatten(),
        Dense(4096, activation = relu),
        Dropout(0.5),
        Dense(4096, activation = relu),
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
