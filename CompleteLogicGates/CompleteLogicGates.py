import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Datapoints: (A, B, Gate)
# Gates: (AND, 0) (OR, 1) (NOR, 2) (NAND, 3) (XOR, 4) (XNOR, 5)
x = [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0],
     [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1],
     [0, 0, 2], [0, 1, 2], [1, 0, 2], [1, 1, 2],
     [0, 0, 3], [0, 1, 3], [1, 0, 3], [1, 1, 3],
     [0, 0, 4], [0, 1, 4], [1, 0, 4], [1, 1, 4],
     [0, 0, 5], [0, 1, 5], [1, 0, 5], [1, 1, 5]]
y = [[0], [0], [0], [1],
     [0], [1], [1], [1],
     [1], [0], [0], [0],
     [1], [1], [1], [0],
     [0], [1], [1], [0],
     [1], [0], [0], [1]]

model = tf.keras.Sequential(layers=[
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dense(64, activation=tf.nn.relu),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(32, activation=tf.nn.relu),
    tf.keras.layers.Dense(16, activation=tf.nn.relu),
    tf.keras.layers.Dense(1, activation=None)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.mae, metrics=['accuracy'])
epochs = 1900
history = model.fit(x, y, epochs=epochs, verbose=1)

plt.plot(range(0, epochs), history.history["loss"], color="b", label="Loss")
plt.plot(range(0, epochs), history.history["accuracy"], color="r", label="Accuracy")
plt.legend()
plt.show()

print(np.rint(model.predict(x)))
