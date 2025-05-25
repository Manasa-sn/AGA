import tensorflow as tf
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Normalize and reshape
x_train = x_train / 255.0
x_test = x_test / 255.0

# Build the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),        # Flatten image
    tf.keras.layers.Dense(256, activation='relu'),        # Hidden layer 1
    tf.keras.layers.Dense(128, activation='relu'),        # Hidden layer 2
    tf.keras.layers.Dense(10)                             # Output layer (logits)
])

# Compile the model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train the model
print("Training...")
model.fit(x_train, y_train, epochs=5, batch_size=64)

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")

# Predict and display 5 test images
logits = model.predict(x_test[:5])
predictions = tf.argmax(logits, axis=1)

for i in range(5):
    plt.imshow(x_test[i], cmap='gray')
    plt.title(f"Predicted: {predictions[i].numpy()}, Actual: {y_test[i]}")
    plt.axis('off')
    plt.show()
