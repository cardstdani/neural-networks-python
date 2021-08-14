x=[]
y=[]

dataLength = 720
testLength = int(dataLength * 0.7)
for i in range(dataLength):
  x.append(i)
  y.append(i*i)

x_train = tf.constant(x[:testLength])
y_train = tf.constant(y[:testLength])

x_test = tf.constant(x[testLength:])
y_test = tf.constant(y[testLength:])

def ac(x):
  return x*x

model = tf.keras.Sequential(layers=[
                                    tf.keras.layers.Dense(100, activation=ac),
                                    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), loss=tf.keras.losses.mse, metrics=["mae", "mse"])
epochs = 440
history = model.fit(x=x_train, y=y_train, epochs=epochs, validation_split=0.3, verbose=1)
model.summary()

plt.plot(range(0, epochs), history.history["loss"], color="b", label="Loss")
plt.figure(2)
plt.plot(range(0, epochs), history.history["mae"], color="r", label="MAE")
plt.figure(3)
plt.plot(range(0, epochs), history.history["mse"], color="g", label="MSE")

#PREDICTIONS VISUALIZATION
plt.figure(4)
plt.plot(x_train, y_train, color="r", label="True values")
plt.plot(x_test, y_test, color="r")
plt.plot(x_train, model.predict(x_train), color="g", label="Predicted values")
plt.plot(x_test, model.predict(x_test), color="g")
plt.legend()
print(model.predict([3000]))
