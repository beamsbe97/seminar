import torch
import torch.nn as nn

import torch
from torch import nn

class PTuning(nn.Module):
    def __init__(self, prompt_length = 20, embed_dim = 1024):

        super().__init__()
        self.prompt_embeddings = nn.Parameter(torch.randn(1, prompt_length, embed_dim))
        self.mlp = HighParamMLP()

    def forward(self, x):
        batch_size = x.size(0)
        prompt = self.mlp(self.prompt_embeddings)
        prompt = prompt.expand(batch_size, -1, -1)  # (batch_size, prompt_length, embed_dim)
        x = torch.cat([prompt,x], dim=1)  # (batch_size, prompt_length + seq_length, embed_dim)

        return x


class HighParamMLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=3072, num_hidden_layers=1, output_dim=1024, dropout=0.1):
        super(HighParamMLP, self).__init__()
        
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)