


resnet_spec = Dict(
                18 => ("basic_block", [2, 2, 2, 2], [64, 64, 128, 256, 512]),
                34 => ("basic_block", [3, 4, 6, 3], [64, 64, 128, 256, 512]),
                50 => ("bottle_neck", [3, 4, 6, 3], [64, 256, 512, 1024, 2048]),
               101 => ("bottle_neck", [3, 4, 23, 3], [64, 256, 512, 1024, 2048]),
               152 => ("bottle_neck", [3, 8, 36, 3], [64, 256, 512, 1024, 2048]))


function _conv3x3(channels, stride, in_channels)
   return Conv2D(channels, (3,3), strides=stride, padding=1,
                    use_bias=false)
end

function BottleneckV2(channels, stride, downsample=false, in_channels=0)
    newchannels = floor(Int, channels/4)
    seq = Sequential(
                BatchNorm(),
                Conv2D(newchannels, (1,1), strides=1, use_bias=False),
                BatchNorm(),
                _conv3x3(newchannels, stride, newchannels),
                BatchNorm(),
                Conv2D(channels, (1,1), strides=1, use_bias=False),
            )

    if downsample
       add(seq, Conv2D(channels, (1,1), stride, use_bias=false))
    else
       self.downsample = None
    end

end

function forward(self::BottleneckV2, F, x)
       residual = x
       x = self.bn1(x)
       act = F.npx.activation if is_np_array() else F.Activation
       x = act(x, act_type='relu')
       if self.downsample:
           residual = self.downsample(x)
       x = self.conv1(x)

       x = self.bn2(x)
       x = act(x, act_type='relu')
       x = self.conv2(x)

       x = self.bn3(x)
       x = act(x, act_type='relu')
       x = self.conv3(x)

       return x + residual
end
