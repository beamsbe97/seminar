import torch
import torch.nn as nn

class Adapter(nn.Module):
    def __init__(self, input_dim, bottleneck_dim=64, activation=nn.ReLU()):
        super(Adapter, self).__init__()
        # 降维
        self.input_dim = input_dim
        self.down_proj = nn.Linear(input_dim, bottleneck_dim)
        # 激活函数
        self.activation = activation
        # 升维
        self.layer_norm = nn.LayerNorm(input_dim)
        self.up_proj = nn.Linear(bottleneck_dim, input_dim)

    def forward(self, x):
        # Adapter的前向传播
        return self.layer_norm(self.up_proj(self.activation(self.down_proj(x)))) + x

from transformers import AutoModel, AutoTokenizer

class AdapterTransformerModel(nn.Module):
    def __init__(self):
        super(AdapterTransformerModel, self).__init__()  # 确保调用父类的__init__方法

        # Adapter 插件：在每一层之后插入 Adapter
        self.encoder_adapters = nn.ModuleList([
            Adapter(input_dim=1024,bottleneck_dim=256)
            for _ in range(24)
        ])
        self.decoder_adapters = nn.ModuleList([
            Adapter(input_dim=512,bottleneck_dim=128)
            for _ in range(8)
        ])
        