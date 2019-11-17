using Photon, LightXML

# This file contains a more elaborate example on training a model on real images.
# This example uses the VOC2007 image dataset. Rather then using segmentation,
# this example uses classification. Without a GPU, this training will be slow!
#
# You can read more about this dataset at:
#    http://host.robots.ox.ac.uk/pascal/VOC/voc2007/
#
# And you can download the dataset from:
#    http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar


# Change location below to where you extracted the VOC2007 dataset
const VOC_DIR = "/home/peter/data/VOCdevkit/VOC2007/"

# Shouldn't be a need to change anything below this line
const IMAGEDIR = joinpath(VOC_DIR,"JPEGImages/")
const LABELDIR = joinpath(VOC_DIR,"Annotations/")
const LABELS = ["chair", "diningtable", "person", "boat", "pottedplant", "car", "sofa",
          "sheep", "cat", "aeroplane", "dog", "train", "bottle", "bird", "bicycle",
          "horse", "cow", "tvmonitor", "motorbike", "bus"]


"""
The labels are stored in XML files, so some parsing is required to retrieve
and transform them into onehot encodings. A single image can have multiple
classes (like it contains a car AND and bicycle).
"""
function get_labels(files::Vector{String}):: Vector{Array}
    labels = []
    for f in files
        bf = f[1:end-4]
        filename = joinpath(LABELDIR, bf * ".xml")

        # Get the class labels for this image. These functions are from
        # the excellent LightXML package
        xroot = root(parse_file(filename))
        objects = get_elements_by_tagname(xroot, "object")
        cls = content.(find_element.(objects, "name"))

        # Now onehot encode the labels
        oh = onehot(unique(cls), LABELS, getContext().dtype)
        push!(labels, oh)
    end
    labels
end

# Get all the image filenames
files = readdir(IMAGEDIR)
@info "Found $(length(files)) image files"

labels = get_labels(files)
@info "Parsed $(length(labels)) label files"

images = joinpath.(IMAGEDIR, files)
data = ImageDataset(images, labels, resize=(200, 200))
data = data |> Normalizer(1.0, 0.5) |> MiniBatch(16)

# Create a simple convolutional network
model = Sequential(
    Conv2D(16, 3, relu),
    Conv2D(64, 3, relu),
    MaxPool2D(),
    Dense(128, relu),
    Dense(20, sigm)
)

workout = Workout(model, MSELoss())

fit!(workout, data, epochs=10)
@info "Finished training for $(workout.epochs) epochs"
