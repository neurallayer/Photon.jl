
# Photon

Photon is a developer friendly framework for Deep Learning in Julia. Under the hood it leverages **KNet** and it provides a **Keras** like API on top of that.

Defining a model couldn't be much simpler and should look familiar if you used
Keras in the past:  

```
model = Sequential(
      Conv2D(16, 3, activation=relu),
      Conv2D(16, 3, activation=relu),
      MaxPool2D(),
      Dense(256, activation=relu),
      Dense()
  )
```


## Features
The goal is to provide a user friendly API for Machine Learning that enables fast prototyping, while remaining fast.

Some of the features:

- Rather then providing all of the input and output shapes, you only provide the output ones. The framework will infer the input sizes the first time it is being invoked.

- If the dimensionality is not matching and there is a sensible default, that will be
applied.


## License
Provided under MIT open source license.


## Todo
Still many things to do:

- Add typing to assist development
- Extend unit test to cover more scenarios
- Implement dataset + dataloader
- Implement more models (resnet,...)
- Write documentation

And btw, we are accepting contributions ;)
