import tensorflow as tf
import numpy as np
import time

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization

model = Sequential([
    Conv2D(16, (3,3), activation="relu", input_shape=(224,224,3)),
    Conv2D(16, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((4,4)),
    Conv2D(64, (3,3), activation="relu"),
    Conv2D(64, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((4,4)),
    Conv2D(256, (3,3), activation="relu"),
    Conv2D(256, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D((4,4)),
    Dense(64, activation="relu"),
    Dense(10, activation="relu")
])


X = np.float32(np.random.randn(16,224,224,3))
model(X)

start=time.time()

for i in range(1,1000):
  model(X)

print("Time spend:", (time.time() - start))
