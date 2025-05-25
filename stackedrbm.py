from sklearn.datasets import load_digits
from sklearn.neural_network import BernoulliRBM
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

# Load digit dataset
digits = load_digits()
X, y = digits.data, digits.target

# Normalize to [0, 1]
X = X / 16.0
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define RBMs
# n_components:  number of hidden units (neurons) in the RBM
# n_iter: number of training iterations (epochs).
rbm1 = BernoulliRBM(n_components=64, learning_rate=0.06, n_iter=20, random_state=0)
rbm2 = BernoulliRBM(n_components=32, learning_rate=0.06, n_iter=20, random_state=0)


# Define classifier
logistic = LogisticRegression(max_iter=1500)
# Stack RBMs + classifier
stacked_rbm = Pipeline(steps=[
    ('rbm1', rbm1),
     ('rbm2', rbm2),
      ('logistic', logistic)
])

# Train the model
stacked_rbm.fit(X_train, y_train)

# Predict on test data
y_pred = stacked_rbm.predict(X_test)

# Print classification report
print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

# Print a comparison of actual vs predicted
print("\n--- Comparison of Actual vs Predicted (First 20 samples) ---")

for i in range(20):
  print(f"Sample {i+1}: Actual = {y_test[i]} | Predicted = {y_pred[i]}")

# Macro avg:	Average of all classes, giving equal weight to each class (good for imbalanced data).
# Weighted avg:	Like macro avg, but gives more importance to classes with more samples.
