import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess MNIST
(x_train, _), _ = tf.keras.datasets.mnist.load_data()
x_train = (x_train.astype('float32') - 127.5) / 127.5
x_train = x_train.reshape(-1, 784)
data = tf.data.Dataset.from_tensor_slices(x_train).shuffle(60000).batch(64)

# Generator and Discriminator
G = tf.keras.Sequential([
  tf.keras.layers.Dense(256, activation='relu', input_shape=(100,)),
  tf.keras.layers.Dense(784, activation='tanh')
])
D = tf.keras.Sequential([
  tf.keras.layers.Dense(256, input_shape=(784,)),
  tf.keras.layers.LeakyReLU(0.2),
  tf.keras.layers.Dense(1, activation='sigmoid')
])
opt_G = tf.keras.optimizers.Adam(0.0002)
opt_D = tf.keras.optimizers.Adam(0.0002)
loss_fn = tf.keras.losses.BinaryCrossentropy()

# Training loop
for epoch in range(5):
  for real in data:
    z = tf.random.normal([real.shape[0], 100])
    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
      fake = G(z)
      D_real = D(real)
      D_fake = D(fake)

      d_loss = loss_fn(tf.ones_like(D_real), D_real) + loss_fn(tf.zeros_like(D_fake), D_fake)
      g_loss = loss_fn(tf.ones_like(D_fake), D_fake)

    grads_D = d_tape.gradient(d_loss, D.trainable_variables)
    grads_G = g_tape.gradient(g_loss, G.trainable_variables)
    opt_D.apply_gradients(zip(grads_D, D.trainable_variables))
    opt_G.apply_gradients(zip(grads_G, G.trainable_variables))
  print(f"Epoch {epoch+1}: D_loss={d_loss:.3f}, G_loss={g_loss:.3f}")

# Generate sample
z = tf.random.normal([1, 100])
img = G(z).numpy().reshape(28, 28)
plt.imshow(img, cmap='gray'); plt.axis('off'); plt.show()
