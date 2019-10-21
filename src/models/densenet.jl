


function _make_dense_block(
    num_layers,
    bn_size,
    growth_rate,
    dropout,
    stage_index,
)
    result = Sequential()
    for _ = 1:num_layers
        add(result, _make_dense_layer(growth_rate, bn_size, dropout))
    end
    return result
end


function _make_dense_layer(growth_rate, bn_size, dropout)
    new_features = Sequential()
    add(new_features, BatchNorm())
    add(new_features, Activation(relu))
    add(new_features, Conv2D(bn_size * growth_rate, (1, 1), use_bias = false))
    add(new_features, BatchNorm())
    add(new_features, Activation(relu))
    add(
        new_features,
        Conv2D(growth_rate, (3, 3), padding = 1, use_bias = false),
    )
    if dropout > 0
        add(new_features, Dropout(dropout))
    end
    out = Concurrent()
    add(out, identity)
    add(out, new_features)
    return out
end


function _make_transition(num_output_features)
    out = Sequential()
    add(out, BatchNorm())
    add(out, Activation(relu))
    add(out, Conv2D(num_output_features, (1, 1), use_bias = false))
    add(out, AvgPool2D(pool_size = 2, strides = 2))
    return out
end


mutable struct DenseNet <: Layer
    features
    output
end

function DenseNet(
    num_init_features,
    growth_rate,
    block_config;
    bn_size = 4,
    dropout = 0,
    classes = 1000,
)

    features = Sequential()
    add(
        features,
        Conv2D(
            num_init_features,
            (7, 7),
            strides = 2,
            padding = 3,
            use_bias = false,
        ),
    )
    add(features, BatchNorm())
    add(features, Activation(relu))
    add(features, MaxPool2D(pool_size = 3, strides = 2, padding = 1))
    # Add dense blocks
    num_features = num_init_features
    for (i, num_layers) in enumerate(block_config)
        add(
            features,
            _make_dense_block(num_layers, bn_size, growth_rate, dropout, i + 1),
        )
        num_features = num_features + num_layers * growth_rate
        if i != length(block_config) - 1
            num_features = floor(Int, num_features / 2)
            add(features, _make_transition(num_features))
        end
    end
    add(features, BatchNorm())
    add(features, Activation(relu))
    add(features, AvgPool2D(pool_size = 7))
    add(features, Flatten())

    output = Dense(classes)
    DenseNet(features, output)
end


function (dn::DenseNet)(x)
    x = dn.features(x)
    dn.output(x)
end

densenet_spec = Dict(
    121 => (64, 32, [6, 12, 24, 16]),
    161 => (96, 48, [6, 12, 36, 24]),
    169 => (64, 32, [6, 12, 32, 32]),
    201 => (64, 32, [6, 12, 48, 32]),
)


DenseNet121(; bn_size = 4, dropout = 0, classes = 1000) =
    DenseNet(densenet_spec[121]..., bn_size = bn_size, dropout=dropout, classes=classes)

DenseNet161(; bn_size = 4, dropout = 0, classes = 1000) =
    DenseNet(densenet_spec[161]..., bn_size = bn_size, dropout=dropout, classes=classes)

DenseNet169(; bn_size = 4, dropout = 0, classes = 1000) =
    DenseNet(densenet_spec[169]..., bn_size = bn_size, dropout=dropout, classes=classes)

DenseNet201(; bn_size = 4, dropout = 0, classes = 1000) =
    DenseNet(densenet_spec[201]..., bn_size = bn_size, dropout=dropout, classes=classes)


@info "Loaded DenseNet models"
