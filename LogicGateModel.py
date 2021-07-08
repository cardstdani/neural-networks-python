import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

#NAND gate emulated with a neural network using tensorflow
x = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
y = np.array([0, 1, 0, 0])

l1 = tf.keras.layers.Dense(units=1, input_shape=[2])
model = tf.keras.Sequential([l1])
model.compile(optimizer=tf.keras.optimizers.SGD(0.1), loss='mean_squared_error')

results = model.fit(x, y, epochs=500)

plt.plot(results.history['loss'])
print(np.round(model.predict([[1, 0]])))
print(l1.get_weights())
