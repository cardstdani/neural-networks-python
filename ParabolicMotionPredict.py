import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import math

x0 = []
y0 = []

for i in range(10000):
  x0.append([random.randint(1, 20), random.randint(1, 89)])
  y0.append((x0[i][0]*math.sin(math.radians(x0[i][1])))/4.905)

x = np.array(x0, dtype=float)
y = np.array(y0, dtype=float)
x = x.reshape(len(x), 2, 1)

model = tf.keras.Sequential([tf.keras.layers.LSTM(units=4, input_shape=(2, 1)),
                             tf.keras.layers.Dense(units=1)])
model.compile(optimizer=tf.keras.optimizers.Adam(0.0001), loss=tf.keras.losses.log_cosh, metrics=['accuracy'])
results = model.fit(x, y, epochs=1000, validation_split=0.1)
model.save("ParabolicMotionModel.h5")

plt.plot(results.history['loss'])
plt.show()
print(model.predict(np.array([[5, 45]]).reshape(1, 2, 1)))
