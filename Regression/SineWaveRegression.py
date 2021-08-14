x=[]
y=[]

dataLength = (360*3)
testLength = int(dataLength * 0.7)
for i in range(-dataLength, dataLength):
  x.append(i)
  y.append(math.sin(math.radians(i)))

x_train = tf.constant(x[:testLength])
y_train = tf.constant(y[:testLength])

x_test = tf.constant(x[testLength:])
y_test = tf.constant(y[testLength:])

def ac(x):
  return tf.math.sin(x)

class Sine(tf.keras.layers.Layer):
    def __init__(self, units=32, input_dim=32):
        super(Sine, self).__init__()
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(initial_value=w_init(shape=(input_dim, units), dtype="float32"), trainable=True)
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(initial_value=b_init(shape=(units,), dtype="float32"), trainable=True)
        a_init = tf.zeros_initializer()
        self.a = tf.Variable(initial_value=a_init(shape=(input_dim, units), dtype="float32"), trainable=True)

    def call(self, inputs):
        return self.a * tf.math.sin(((tf.cast(inputs, dtype=tf.float32) * self.w) + self.b))

    def get_config(self):
        config = super(Sine, self).get_config()
        config.update({"k": self.k})
        return config

tf.random.set_seed(42)
model = tf.keras.Sequential(layers=[
                                    Sine(1, 1),
                                    tf.keras.layers.Dense(1)
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.mse, metrics=["mae", "mse"])
epochs = 500
history = model.fit(x=x_train, y=y_train, epochs=epochs, validation_split=0.3, verbose=1)
model.summary()

#print(x_train, y_train)
plt.plot(range(0, epochs), history.history["loss"], color="b", label="Loss")
plt.figure(2)
plt.plot(range(0, epochs), history.history["mae"], color="r", label="MAE")
plt.figure(3)
plt.plot(range(0, epochs), history.history["mse"], color="g", label="MSE")

#EVALUATE THE MODEL
plt.figure(figsize=(10, 10), dpi=80)
plt.plot(x_train, y_train, color="r", label="True values")
plt.plot(x_test, y_test, color="r")
plt.plot(x_train, model.predict(x_train), color="g", label="Predicted values")
plt.plot(x_test, model.predict(x_test), color="g")
plt.legend()
print(model.predict([30]))
print(model.predict([30+(360*8)]))
print(model.predict([30+(360*18)]))
print(model.predict([30+(360*280)]))
