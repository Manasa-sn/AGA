#importing library
import numpy as np
import matplotlib.pyplot as plt
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.datasets import mnist

#Loading the dataset
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# reshape in the input data for the model
x_train = x_train.reshape(len(x_train), 28, 28, 1)
x_test = x_test.reshape(len(x_test), 28, 28, 1)
x_test.shape

#model implementation
model = Sequential([
    # encoder network
    Conv2D(32, 3, activation='relu', padding='same', input_shape=(28, 28, 1)),
    MaxPooling2D(2, padding='same'),
    Conv2D(16, 3, activation='relu', padding='same'),
    MaxPooling2D(2, padding='same'),
    # decoder network
    Conv2D(16, 3, activation='relu', padding='same'),
    UpSampling2D(2),
    Conv2D(32, 3, activation='relu', padding='same'),
    UpSampling2D(2),
    # output layer
    Conv2D(1, 3, activation='sigmoid', padding='same')
])

model.compile(optimizer='adam', loss='binary_crossentropy')
model.fit(x_train, x_train, epochs=20, batch_size=256, validation_data=(x_test, x_test))

#storing the predected output here and visualizing the result
pred = model.predict(x_test)

#Visual Representation
index = np.random.randint(len(x_test))
plt.figure(figsize=(10, 4))

# display original image
ax = plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(x_test[index].reshape(28,28))
plt.gray()

# display compressed image
ax = plt.subplot(1, 2, 2)
plt.title("compressed Image")
plt.imshow(pred[index].reshape(28,28))
plt.gray()
plt.show()

from sklearn.metrics import mean_squared_error

# Get original and predicted images
original = x_test[index].reshape(28, 28)
reconstructed = pred[index].reshape(28, 28)

# Compute Mean Squared Error
mse = mean_squared_error(original, reconstructed)
print(f"Mean Squared Error (MSE) between original and reconstructed image: {mse}")
