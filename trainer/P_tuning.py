import torch
import torch.nn as nn

import torch
from torch import nn

class PTuning(nn.Module):
    def __init__(self, prompt_length = 20, embed_dim = 1024):
        """
        :param base_model: 原始 Transformer/MAE 模型
        :param prompt_length: 软提示向量的长度 P
        :param embed_dim: 嵌入维度 D
        """
        super().__init__()
        self.prompt_embeddings = nn.Parameter(torch.randn(1, prompt_length, embed_dim))
        self.mlp = HighParamMLP()

    def forward(self, x):
        """
        :param x: 输入序列 (batch_size, seq_length, embed_dim)
        :return: 模型输出
        """
        # 将软提示向量添加到输入序列的开头
        batch_size = x.size(0)
        prompt = self.mlp(self.prompt_embeddings)
        prompt = prompt.expand(batch_size, -1, -1)  # (batch_size, prompt_length, embed_dim)
        x = torch.cat([prompt,x], dim=1)  # (batch_size, prompt_length + seq_length, embed_dim)

        # 通过模型前向传播
        return x


class HighParamMLP(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=3072, num_hidden_layers=1, output_dim=1024, dropout=0.1):
        """
        :param input_dim: 输入特征维度
        :param hidden_dim: 隐藏层维度（越大参数量越高）
        :param num_hidden_layers: 隐藏层数量
        :param output_dim: 输出特征维度
        :param dropout: Dropout 比例
        """
        super(HighParamMLP, self).__init__()
        
        # 输入层
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        
        # 隐藏层
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)