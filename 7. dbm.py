import numpy as np

# Sigmoid activation
def sigmoid(x):
  return 1 / (1 + np.exp(-x))

# Sampling binary units based on probabilities
def sample(prob):
  return np.random.binomial(1, prob)

# Sampling function for a layer
def sample_layer(input_data, weights, bias):
  activation = np.dot(input_data, weights) + bias
  prob = sigmoid(activation)
  return sample(prob), prob

# One training step for a simplified DBM
def dbm_step(v0, W1, b1, W2, b2, lr=0.01):
  # ======== UPWARD PASS ========
  h1, h1_prob = sample_layer(v0, W1, b1) # From visible to hidden1
  h2, h2_prob = sample_layer(h1, W2, b2) # From hidden1 to hidden2
  # ======== DOWNWARD PASS (Reconstruction) ========
  h1_down, _ = sample_layer(h2, W2.T, np.zeros_like(b1)) # Reconstruct hidden1
  v1, _ = sample_layer(h1_down, W1.T, np.zeros_like(v0)) # Reconstruct visible
  
  # ======== WEIGHT & BIAS UPDATES (Contrastive Divergence-like) ========
  # Positive phase
  pos_W1 = np.outer(v0, h1)
  pos_W2 = np.outer(h1, h2)

  # Negative phase
  neg_W1 = np.outer(v1, h1_down)
  neg_W2 = np.outer(h1_down, h2)

  # Update weights and biases
  W1 += lr * (pos_W1 - neg_W1)
  W2 += lr * (pos_W2 - neg_W2)
  b1 += lr * (h1 - h1_down)
  b2 += lr * (h2 - h2_prob)
  return W1, b1, W2, b2

# ======== INITIALIZATION ========
np.random.seed(42) # For reproducibility
v0 = np.array([1, 0, 1, 0]) # 4 visible units (input)
W1 = np.random.randn(4, 3) * 0.1 # 4 ↔ 3 weights (visible ↔ hidden1)
b1 = np.zeros(3)
W2 = np.random.randn(3, 2) * 0.1 # 3 ↔ 2 weights (hidden1 ↔ hidden2)
b2 = np.zeros(2)

# ======== TRAINING STEP ========
W1, b1, W2, b2 = dbm_step(v0, W1, b1, W2, b2)

# ======== OUTPUT ========
print("Updated W1 (v ↔ h1):\n", W1)
print("Updated b1 (h1):", b1)
print("Updated W2 (h1 ↔ h2):\n", W2)
print("Updated b2 (h2):", b2)
