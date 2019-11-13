using Photon, LightXML

const VOC_DIR = "/home/peter/data/VOCdevkit/VOC2007/"
const IMAGEDIR = joinpath(VOC_DIR,"JPEGImages/")
const LABELDIR = joinpath(VOC_DIR,"Annotations/")
const LABELS = ["chair", "diningtable", "person", "boat", "pottedplant", "car", "sofa",
          "sheep", "cat", "aeroplane", "dog", "train", "bottle", "bird", "bicycle",
          "horse", "cow", "tvmonitor", "motorbike", "bus"]

# Get all the image filenames
images = readdir(IMAGEDIR)

"""
The labels are stored in XML files, so some parsing is required to retrieve
and transform them into onehot encodings. A single image can have multiple
classes (like it contains a car AND and bicycle).
"""
function get_labels(images)
    labels = []
    for f in images
        bf = f[1:end-4]
        filename = joinpath(LABELDIR, bf * ".xml")
        xroot = root(parse_file(filename))
        objects = get_elements_by_tagname(xroot, "object")
        cls = content.(find_element.(objects, "name"))
        oh = onehot(unique(cls), LABELS, getContext().dtype)
        push!(labels, oh)
    end
    labels
end

labels = get_labels(images)
images = joinpath.(IMAGEDIR, images)

ds = ImageDataset(images, labels, resize=(200, 200))

dl = Dataloader(ds, 16)

# We create a simple convolutional network
model = Sequential(
    Conv2D(16, 3, relu),
    Conv2D(64, 3, relu),
    MaxPool2D(),
    Dense(128, relu),
    Dense(20, sigm)
)

workout = Workout(model, mse)

fit!(workout, dl, epochs=10)
