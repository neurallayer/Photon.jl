using Photon, LightXML

VOC_IMAGEDIR = "/home/peter/data/VOCdevkit/VOC2007/JPEGImages/"
VOC_LABELDIR = "/home/peter/data/VOCdevkit/VOC2007/Annotations/"
LABELS = ["chair", "diningtable", "person", "boat", "pottedplant", "car", "sofa",
          "sheep", "cat", "aeroplane", "dog", "train", "bottle", "bird", "bicycle",
          "horse", "cow", "tvmonitor", "motorbike", "bus"]


images = readdir(VOC_IMAGEDIR)

"""
Simple onehot encoder for a single sample.
"""
struct OneHotEncoder
    labels
    dtype
    OneHotEncoder(labels; dtype=getContext().dtype) = new(labels,dtype)
end

function (oh::OneHotEncoder)(x::Any)
      result = zeros(oh.dtype, length(oh.labels))
      result[findfirst(x .== oh.labels)] = 1
      result
end

function (oh::OneHotEncoder)(X::AbstractArray)
      result = zeros(oh.dtype, length(oh.labels))
      for x in X
          result[findfirst(x .== oh.labels)] = 1
      end
      result
end


encoder = OneHotEncoder(LABELS)


labelfiles = readdir(VOC_LABELDIR)
file_label= Dict()
for f in labelfiles
    xroot = root(parse_file(VOC_LABELDIR * f))
    objects = get_elements_by_tagname(xroot, "object")
    cls = content.(find_element.(objects, "name"))
    file_label[f[1:end-4]] = encoder(unique(cls))
end

file_label
# hot_labels = unique(vcat(values(labels)...))


labels = []
for image in images
    base = image[1:end-4]
    label = file_label[base]
    push!(labels, label)
end


images = VOC_IMAGEDIR .* images

ds = ImageDataset(images, labels, resize=(100,100))

dl = Dataloader(ds,16)

model = Sequential(
    Conv2D(16, 3, relu),
    Conv2D(64, 3, relu),
    MaxPool2D(),
    Dense(512, relu),
    Dense(20, sigm)
)

workout = Workout(model, mse)

fit!(workout, dl, epochs=10)
