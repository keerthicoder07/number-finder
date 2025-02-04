# model_trainer.py  
import numpy as np  
import tensorflow as tf  
from tensorflow import keras  
from tensorflow.keras import layers  

# Load the MNIST dataset  
mnist = keras.datasets.mnist  
(x_train, y_train), (x_test, y_test) = mnist.load_data()  

# Preprocess the dataset  
x_train = x_train.astype("float32") / 255.0  
x_test = x_test.astype("float32") / 255.0  

# Build the model  
model = keras.Sequential([  
    layers.Flatten(input_shape=(28, 28)),  
    layers.Dense(128, activation='relu'),  
    layers.Dense(10, activation='softmax')  
])  

# Compile the model  
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])  

# Train the model  
model.fit(x_train, y_train, epochs=5)  

# Save the model  
model.save("mnist_model.h5")