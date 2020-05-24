
<p align="center">
<img width="400px" src="https://raw.githubusercontent.com/neurallayer/Photon.jl/master/docs/src/assets/logo.png"/>
</p>


[![License](http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat)](LICENSE.md)
[![Build Status](https://travis-ci.org/neurallayer/Photon.jl.svg?branch=master)](https://travis-ci.org/neurallayer/Photon.jl)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://neurallayer.github.io/Photon.jl/dev/)
[![codecov](https://codecov.io/gh/neurallayer/Photon.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/neurallayer/Photon.jl)
[![Join the julia slack](https://img.shields.io/badge/chat-slack%23photon-yellow.svg)](https://slackinvite.julialang.org)


**Photon** is a developer friendly framework for Deep Learning in Julia. Under the hood
it leverages **Knet** and it provides a Keras like API on top of that.


## Installation
You can install Photon as any othe rpackage in Julia. From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add Photon
```

You can also isntall it from the master branch:

```
pkg> add https://github.com/neurallayer/Photon.jl
```

## Documentation

Click [here](https://neurallayer.github.io/Photon.jl/dev/) to go to the documentation site.


## Usage
Defining a model is straightforward and should look familiar if you used Keras
or MXnet in the past:

A two layers fully connected network:

```julia
model = Sequential(
      Dense(256, relu),
      Dense(10)
  )
```

A convolutional network with maxpooling (note that Photon takes care
of the flattening so you can connect a Dense layer directly to a convolutional
layer):

```julia
model = Sequential(
      Conv2D(16, 3, relu),
      Conv2D(16, 3, relu),
      MaxPool2D(),
      Dense(256, relu),
      Dense(10)
  )
```

Or a recurrent LSTM network:

```julia
model = Sequential(
      LSTM(256, 3),
      Dense(64, relu),
      Dense(10)
  )
```

And also the training of the model can be done through an
easy API:

```julia
workout = Workout(model, some_loss_function)
train!(workout, data, epochs=10)
```

## Performance
The combination of Deep Learning and Julia is a very performant one. Especially
when running the models on a GPU. Some tests reveal **speedups of up to 100%** when
compared one of the most popular framework Keras/Tensorflow 2.0.

The code that has been used to test is available in the *performance* sub directory.

## Features
The main goal is to provide a user friendly API for Machine Learning that enables
developing both prototype and production ready solutions while remaining fast.

Some of the key features:

- Good support for the various ways you can to retrieve and transform your data.

- Develop models with a minimal amount of code.

- Get all the required insights and visualizations into the performance of the model.


## Todo
There remain many things to do.

- Extend unit tests to cover more code (> 90%)
- Implement more models including trained weights (resnet,...)
- Write tutorials and improve code documentation
- Finalise the API's so we can release 1.0
- Implement more complex building blocks like transformers
- Additional loss functions

And b.t.w, we are always open in accepting contributions ;)

## License
Photon is provided under the MIT open source license.


## References
We used several other open source frameworks for code and inspiration

- Knet (pronounced "kay-net") is the Koç University deep learning framework
  implemented in Julia by Deniz Yuret and collaborators. It is the
  back-end for Photon, partially due to its excellent performance on GPU's.

- Flux, we use it for inspiration. This has to be one of the most
  beautiful code bases out there.

- Keras and MXNet for their well thought out API's. Also copied some of their
  excellent documentation for layers and losses.

- And of course Julia, that enables writing very fast deep learning applications.
