import numpy as np
import torch
import torch.nn as nn


#Линейный слой на NumPy
class Linear_Layer_np:
    def __init__(self, n_features: int, m_features: int):
        self.n_features = n_features
        self.m_features = m_features

        self.weights = np.random.rand(m_features, n_features)
        self.bias = np.random.rand(m_features)

    def forward(self, x):
        return np.dot(x, self.weights.T) + self.bias

    def __call__(self, x):
        return  self.forward(x)

layer = Linear_Layer_np(3, 5)
input_data = np.random.randn(5, 3)
print(input_data)
print("-" * 50)
output_np = layer(input_data)
print(output_np)
print("-" * 50)

#Линейный слой на PyTorch
class Linear_Layer_pt:
    def __init__(self, n_features, m_features):
        self.n_features = n_features
        self.m_features = m_features

        self.weights = torch.randn(m_features, n_features)
        self.bias = torch.randn(m_features)

    def forward(self, x):
        return torch.matmul(x, self.weights.T) + self.bias

    def __call__(self, x):
        return self.forward(x)

layer = Linear_Layer_pt(3, 5)
input_data = torch.randn(5, 3)
print(input_data)
print("-" * 50)
output_pt = layer(input_data)
print(output_pt)
print("-" * 50)

#ReLU на NumPy
class ReLU_np:
    def __init__(self):
        pass

    def forward(self, x):
        return np.maximum(0, x)

    def __call__(self, x):
        return self.forward(x)

relu = ReLU_np()
act_output_np = relu(output_np)
print(act_output_np)
print("-" * 50)

#ReLU на PyTorch
class ReLU_pt:
    def __init__(self):
        pass

    def forward(self, x):
        return torch.maximum(torch.tensor(0), x)

    def __call__(self, x):
        return  self.forward(x)

relu = ReLU_pt()
act_output_pt = relu(output_pt)
print(act_output_pt)
print("-" * 50)

#SoftMax на NumPy
class SoftMax_np:
    def __init__(self):
        pass

    def forward(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def __call__(self, x):
        return self.forward(x)

softmax = SoftMax_np()
prob_softmax_np = softmax(act_output_np)
print(prob_softmax_np)
print("-" * 50)

#SoftMax на PyTorch
class SoftMax_pt:
    def __init__(self):
        pass

    def forward(self, x):
        exp_x = torch.exp(x)
        return exp_x/torch.sum(exp_x)

    def __call__(self, x):
        return self.forward(x)

softmax = SoftMax_pt()
prob_softmax_pt = softmax(act_output_pt)
print(prob_softmax_pt)
print("-" * 50)

# Self-Attention на NumPy
class Self_Attention_np:
    def __init__(self, d_vec):
        self.d_vec = d_vec

        self.Wq = np.random.randn(d_vec, d_vec)
        self.Wk = np.random.randn(d_vec, d_vec)
        self.Wv = np.random.randn(d_vec, d_vec)

    def forward(self, x):
        Q = np.matmul(x, self.Wq)
        K = np.matmul(x, self.Wk)
        V = np.matmul(x, self.Wv)

        attention_scores = np.matmul(Q, K.transpose(0, 2, 1)) / np.sqrt(self.d_vec)

        attention_weight = np.matmul(np.exp(attention_scores - np.max(attention_scores))/np.sum(np.exp(attention_scores)), V)

        return attention_weight

    def __call__(self, x):
        return self.forward(x)

batch_size, seq_len, embed_dim = 2, 4, 8
x = np.random.randn(batch_size, seq_len, embed_dim)

self_attention = Self_Attention_np(embed_dim)
output_np = self_attention(x)
print(output_pt)
print("-" * 50)

#Self-Attentiom на PyTorch
class Self_Attention_pt:
    def __init__(self, d_vec):
        self.d_vec = d_vec

        self.Wq = torch.randn(d_vec, d_vec)
        self.Wk = torch.randn(d_vec, d_vec)
        self.Wv = torch.randn(d_vec, d_vec)

    def forward(self, x):
        Q = torch.matmul(x, self.Wq)
        K = torch.matmul(x, self.Wk)
        V = torch.matmul(x, self.Wv)

        attention_scores = torch.nn.functional.softmax((Q @ K.transpose(-2, -1) / torch.sqrt(torch.tensor(self.d_vec))), dim=-1)
        attention_weights = torch.matmul(attention_scores, V)

        return  attention_weights

    def __call__(self, x):
        return self.forward(x)

batch_size, seq_len, embed_dim = 2, 4, 8
x = torch.randn(batch_size, seq_len, embed_dim)

self_attention = Self_Attention_pt(embed_dim)
output_pt = self_attention(x)
print(output_pt)
print("-" * 50)

class DynamicTanh(nn.Module):
    def __init__(self, dim, init_alpha=0.5):
        super().__init__()
        self.dim = dim
        self.alpha = nn.Parameter(torch.ones(1) * self.alpha)
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        x = torch.tanh(self.alpha * x)
        x = self.gamma * x + self.beta
        return x
    
    