
# Photon

[![Build Status](https://travis-ci.org/neurallayer/Photon.jl.svg?branch=master)](https://travis-ci.org/neurallayer/Photon.jl)
[![codecov](https://codecov.io/gh/neurallayer/Photon.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/neurallayer/Photon.jl)

Photon is a developer friendly framework for Deep Learning in Julia. Under the hood it leverages **Knet** and it provides a **Keras** like API on top of that.

It is right now still very much alpha quality and the main goal of the current version is to see what API's work best. So expect still to see some changes in the API the upcoming releases.

## Installation

```julia
julia> Pkg.add("Photon")
```


## Usage
Defining a model couldn't be much simpler and should look familiar if you used Keras in the past:  

A two layers fully connected network:

```julia
model = Sequential(
      Dense(256, activation=relu),
      Dense(10)
  )
```

A convolutional network:

```julia
model = Sequential(
      Conv2D(16, 3, activation=relu),
      Conv2D(16, 3, activation=relu),
      MaxPool2D(),
      Dense(256, activation=relu),
      Dense(10)
  )
```

Or a recurrent LSTM network:

```julia
model = Sequential(
      LSTM(256, 3),
      Dense(64, activation=relu),
      Dense(10)
  )
```

And also the training of the model can be done through an
easy API:

```julia
workout = Workout(model, nll, ADAM())
fit!(workout, data, epochs=50)
```

## Perfomance
The combination of Deep Learning and Julia is a very performant one. Especially
when running the models on a GPU. Some tests reveal **speedups of up to 100%** when
compared one of the most popular framework Keras/Tensorflow 2.0.

Some early results based on running models on an Intel 7820X machine with a
GTX 1080TI NVidia graphics card installed and running Ubuntu 18.04:

|Scenario | Photon | Keras | Flux |
| :---    |   ---: |  ---: | ---: |
| Conv predict | 5.7 | 12.6 | 11.2 |
| Conv train | todo | todo | todo |
| LSTM predict | todo | todo | todo |
| LSTM train | todo | todo | todo |

The code that has been used is available in the performance directory subdirectory.

## Features
The goal is to provide a user friendly API for Machine Learning that enables
both prototyping and production ready solutions, while remaining fast.

Some of the features:

- The framework will infer the input sizes the first time it is being invoked. This
  makes it quicker to get started, but also making the layers more reusable.

- Where possible, sensible defaults are selected.

- Make it easy to create reproducable results.

## Todo
This software is still alpha quality and there remain many things to do:

- Add typing to assist the compiler development
- Extend unit tests to cover more scenarios
- Implement dataset + dataloader
- Implement more models (resnet,...)
- Write documentation
- Finalize Workout API

And b.t.w, we are always open in accepting contributions ;)

## License
Photon is provided under the MIT open source license.


## References
We also used several other open source frameworks for code and inspiration

- Knet (pronounced "kay-net") is the Koç University deep learning framework
  implemented in Julia by Deniz Yuret and collaborators. It is right now the backend
  for Photon due to its excellent performance on GPU's.

- FluxML, we used some of their optimise code. This has to be one of the more
  beautiful code bases out there.

- Keras and MXNet for their well thought out API's.

- And of course Julia, that enables writing very fast data applications
