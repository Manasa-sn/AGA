# Datasets: https://www.kaggle.com/datasets/imkushwaha/movielens-100k-dataset?select=ua.test
# Datasets: https://www.kaggle.com/datasets/ashukr/movie-rating-data

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# Load datasets
movies = pd.read_csv("movie.csv", sep='::', header=None, engine='python', encoding='latin-1')
users = pd.read_csv("u.user", sep='::', header=None, engine='python', encoding='latin-1')
ratings = pd.read_csv("rating.csv", sep='::', header=None, engine='python', encoding='latin-1')

training_set = pd.read_csv("ua.base", delimiter='\t')
training_set = np.array(training_set, dtype='int')
test_set = pd.read_csv("ua.test", delimiter='\t')
test_set = np.array(test_set, dtype='int')

nb_users = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nb_movies = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

# Convert data to user-movie matrix
def convert(data):
    new_data = []
    for id_users in range(1, nb_users + 1):
        id_movies = data[:, 1][data[:, 0] == id_users]
        id_ratings = data[:, 2][data[:, 0] == id_users]
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

training_set = convert(training_set)
test_set = convert(test_set)

# Convert to Torch tensors
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

# Binarize the data
training_set[training_set == 0] = -1
training_set[training_set == 1] = 0
training_set[training_set == 2] = 0
training_set[training_set >= 3] = 1
test_set[test_set == 0] = -1
test_set[test_set == 1] = 0
test_set[test_set == 2] = 0
test_set[test_set >= 3] = 1

# RBM class
class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)  # Hidden bias
        self.b = torch.randn(1, nv)  # Visible bias

    def sample_h(self, x):
        wx = torch.mm(x, self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, y):
        wy = torch.mm(y, self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(ph0.t(), v0) - torch.mm(phk.t(), vk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

# Training
nv = len(training_set[0])
nh = 100
batch_size = 100
rbm = RBM(nv, nh)
nb_epoch = 10

for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.0
    for id_user in range(0, nb_users - batch_size, batch_size):
        vk = training_set[id_user:id_user+batch_size]
        v0 = training_set[id_user:id_user+batch_size]
        ph0, _ = rbm.sample_h(v0)
        for k in range(10):
            _, hk = rbm.sample_h(vk)
            _, vk = rbm.sample_v(hk)
            vk[v0 < 0] = v0[v0 < 0]
        phk, _ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0 >= 0] - vk[v0 >= 0]))
        s += 1
    print(f'Epoch: {epoch} | Loss: {train_loss / s:.4f}')

# Testing
test_loss = 0
s = 0.0
for id_user in range(nb_users):
    v = training_set[id_user:id_user+1]
    vt = test_set[id_user:id_user+1]
    if len(vt[vt >= 0]) > 0:
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt >= 0] - v[vt >= 0]))
        s += 1
print(f'Test Loss: {test_loss / s:.4f}')
