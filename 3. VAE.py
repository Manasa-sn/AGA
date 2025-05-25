import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.datasets import mnist

# Hyperparameter
batch_size = 100
original_dim = 784
latent_dim = 2
intermediate_dim = 256
epochs = 5

# Sampling Layer
class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        # epsilon with the same shape as of z_mean by normal distribution(mean=0, var=1)
        # This is the source of randomness — noise sampled from N(0,1).
        epsilon = tf.random.normal(shape=tf.shape(z_mean))

        # tf.exp(0.5 * z_log_var) computes the standard deviation of the latent distribution.
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Encoder
inputs = layers.Input(shape=(original_dim,))
h = layers.Dense(256, activation='relu')(inputs)
z_mean = layers.Dense(2, name="z_mean")(h)
z_log_var = layers.Dense(2, name="z_log_var")(h)
z = Sampling()([z_mean, z_log_var])
encoder = Model(inputs, [z_mean, z_log_var, z], name="encoder")

# Decoder
latent_inputs = layers.Input(shape=(2,))
x = layers.Dense(256, activation='relu')(latent_inputs)
outputs = layers.Dense(original_dim, activation='sigmoid')(x)
decoder = Model(latent_inputs, outputs, name="decoder")

# VAE as subclassed model
class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]

        # tf.GradientTape() is a TensorFlow tool that records operations for automatic differentiation 
        # — meaning it tracks computations so it can calculate gradients later.
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_sum(
                tf.keras.losses.binary_crossentropy(data, reconstruction), axis=-1
            )
            kl_loss = -0.5 * tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1
            )
            total_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return {"loss": total_loss}

vae = VAE(encoder, decoder)
vae.compile(optimizer='rmsprop')

# Load data
(x_train, _), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.
x_train = x_train.reshape((-1, original_dim))
x_test = x_test.reshape((-1, original_dim))

# Train
vae.fit(x_train, x_train, epochs=epochs, batch_size=batch_size)

# Encode test data to latent space
z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
plt.figure(figsize=(6, 6))
plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
plt.colorbar()
plt.show()

# Generate digits from latent space
n = 15
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
grid_x = norm.ppf(np.linspace(0.05, 0.95, n))
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(digit_size, digit_size)
        figure[i * digit_size: (i + 1) * digit_size,
               j * digit_size: (j + 1) * digit_size] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure, cmap='Greys_r')
plt.show()
