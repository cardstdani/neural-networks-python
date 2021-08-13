import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

x=[]
y=[]

def getHeight(y, v):
  return y + ((v**2)/9.81) - ((v**2)/(2*9.81))

for i in range(100000):
  vector = [random.randint(0, 50), random.randint(0, 50)]  
  x.append(vector)
  y.append(getHeight(vector[0], vector[1]))

x_train = tf.constant(x)
y_train = tf.constant(y)

model = tf.keras.Sequential(layers=[#tf.keras.layers.Dense(2, input_shape=(None,2), activation=tf.keras.activations.relu),
                                    tf.keras.layers.Dense(100, activation=tf.keras.activations.selu),
                                    tf.keras.layers.Dense(100, activation=tf.keras.activations.selu),
                                    tf.keras.layers.Dense(100, activation=tf.keras.activations.selu),
                                    tf.keras.layers.Dense(1, activation=None)
                                    ])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.mae, metrics=['mae'])
model.fit(x=x_train, y=y_train, epochs=200, validation_split=0.1, verbose=True)

#print(x_train, y_train)
print(model.predict([[0, 45]]))
