import torch
import torch.nn as nn
from functools import partial
import sys
import models.models_mae as models_mae
import os
from PIL import Image
import torchvision.transforms as T
import h5py
import torchvision.transforms.functional as TF
from PIL import Image
from omegaconf import OmegaConf
import torch.nn.functional as F

def gaussian_weight_matrix(size, center, sigma=1.0):
    """
    Create a 2D Gaussian weight matrix with the specified size and sigma, centered at the given location.
    """
    x = torch.linspace(0, size - 1, size)
    y = torch.linspace(0, size - 1, size)
    xx, yy = torch.meshgrid(x, y, indexing='ij')
    kernel = torch.exp(-((xx - center[0])**2 + (yy - center[1])**2) / (2 * sigma**2))
    return kernel / kernel.sum()

class Matrix():
    def __init__(self,sigma,device):
        self.gw = [[] for i in range(7)]
        for i in range(7):
            for j in range(7):
                self.gw[i].append(gaussian_weight_matrix(7, center=(i, j), sigma=sigma).to(device=device))
    def calculate_attention(self,key, value1, value2, query, dropout, tsigma=1.0):
        B, N, _, _, D = key.size()
        _, h, w, _ = query.size()

        # Reshape query to 49 x B x D
        query_reshaped = query.view(B, -1, D).permute(1, 0, 2)

        list1 = []
        list2 = []
        key = key.to(torch.float32)
            
        for i in range(h * w):
            # Calculate the position of the current patch
            patch_y, patch_x = divmod(i, w)
            
            # Current query vector
            q = query_reshaped[i]  # B x D
            # print("q min:", q.min().item(), "q max:", q.max().item())        
            # print("key min:", key.min().item(), "key max:", key.max().item())        
            q = q.to(torch.float32)
            # Compute similarity scores between query and key
            score = torch.einsum('bd,bnwhd->bnwh', q, key)  # B x N x 7 x 7

            # Create a Gaussian weight matrix centered at the current patch position
            gaussian_matrix = self.gw[patch_y][patch_x]
            
            # Apply Gaussian weighting
            score = score * gaussian_matrix
            # print("score min:", score.min().item(), "score max:", score.max().item())        
            # Reshape and apply softmax
            attn_weight = F.softmax(score.view(B, -1), dim=-1).view(B, N, 7, 7)
            # attn_weight = dropout(attn_weight)
            # print("attn_weight min:", attn_weight.min().item(), "attn_weight max:", attn_weight.max().item())        

            # Compute weighted sum
            attn_output1 = torch.einsum('bnwh,bnwhd->bd', attn_weight, value1)
            attn_output2 = torch.einsum('bnwh,bnwhd->bd', attn_weight, value2)
            # Collect the result
            list1.append(attn_output1)
            list2.append(attn_output2)

        output1 = torch.stack(list1,dim=1)
        output2 = torch.stack(list2,dim=1)

        return output1, output2

class PromptGenerator(nn.Module):
    def __init__(self,dropout = 0,sigma = 1.0,device='cpu'):
        super().__init__()
        #self.CrossAttention_S = nn.MultiheadAttention(embed_dim = 1024, dropout = dropout,num_heads = 8,batch_first=True)
        self.SelfAttention_Q = nn.MultiheadAttention(embed_dim = 1024, dropout = dropout, num_heads = 8,batch_first=True)
        self.SelfAttention_S = nn.MultiheadAttention(embed_dim = 1024, dropout = dropout, num_heads = 8,batch_first=True)
        print('dropout ',dropout)
        self.Layer_norm = nn.LayerNorm(1024)
        self.Linearq = nn.Linear(1024,1024)
        self.Lineark = nn.Linear(1024,1024)
        self.Linearv1 = nn.Linear(1024,1024)
        self.Linearv2 = nn.Linear(1024,1024)
        self.dropout = nn.Dropout(0.25)
        self.Matrix = Matrix(sigma,device=device)
        self.sigma = sigma
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)
        nn.init.eye_(self.Linearv2.weight)
        nn.init.zeros_(self.Linearv2.bias)

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # support_features  [B,N,98,1024]
    # query_features    [B,1,98,1024]
    # first self_attention 
    #       input       [B*N,98,1024]
    # second self_attention
    #       input       [B,49,1024]
    # third cross_attention
    #       input  k,v  [B*49,N,1024]
    #       input  q    [B*49,1,1024]
    #       get_attnweight v2 @ weight
    # fourth return     [B,196,1024]
    def forward(self, support_features, query_features):
        batchsize = support_features.shape[0]
        N = support_features.shape[1] 
        # print(support_features.shape)
        qss = support_features.reshape(batchsize*N,98,1024)
        qss_layer_norm = self.Layer_norm(qss)
        ats_ans,_ = self.SelfAttention_S(qss_layer_norm,qss_layer_norm,qss_layer_norm)
        support_features = qss + ats_ans #[B*N,98,1024]

        ## support's self_attention to register

        query_features = query_features.reshape(batchsize,1,7,14,1024)
        query_features_img = query_features[:,:,:,:7,:]
        query_features_mask = query_features[:,:,:,7:,:]
        query_features_img = query_features_img.reshape(batchsize,49,1024)
        qsq_layner_norm = self.Layer_norm(query_features_img)
        atq_ans,_ = self.SelfAttention_Q(qsq_layner_norm,qsq_layner_norm,qsq_layner_norm)
        query_features_img = query_features_img + atq_ans #[B,49,1024]

        ## query's img self_attention to register

        query_features_img = query_features_img.reshape(batchsize,7,7,1024)
        support_features = support_features.reshape(batchsize*N,7,14,1024)
        support_features_img = support_features[:,:,:7,:]
        support_features_mask = support_features[:,:,7:,:]
        support_features_img = support_features_img.reshape(batchsize,N,7,7,1024)
        support_features_mask = support_features_mask.reshape(batchsize,N,7,7,1024)
        # support_features_img = support_features_img.permute(0,2,1,3).reshape(batchsize*49,N,1024)
        # support_features_mask = support_features_mask.permute(0,2,1,3).reshape(batchsize*49,N,1024)
        # print("support_features min:", support_features.min().item(), "support_features max:", support_features.max().item())        
        # print("query_features_img min:", query_features_img.min().item(), "query_features_img max:", query_features_img.max().item())        

        ## update this shape matching

        #attn_out1,attn_weight = self.CrossAttention_S(query_features_img,support_features_img,support_features_img)         #[B*49,1,1024]
        #attn_out2 = (attn_weight @ (self.Linear(support_features_mask)))          #[B*49,1,1024]


        attn_out1,attn_out2 = self.Matrix.calculate_attention(self.Lineark(support_features_img),self.Linearv1(support_features_img),self.Linearv2(support_features_mask),self.Linearq(query_features_img),self.dropout,tsigma=self.sigma)

        ##after attention process the answer
        # print("attn_out1 min:", attn_out1.min().item(), "attn_out1 max:", attn_out1.max().item())        
        # print("attn_out2 min:", attn_out2.min().item(), "attn_out2 max:", attn_out2.max().item())        

        attn_out1 = attn_out1.reshape(batchsize,7,7,1024)
        attn_out2 = attn_out2.reshape(batchsize,7,7,1024)
        query_features_img = query_features_img.reshape(batchsize,7,7,1024)
        query_features_mask = query_features_mask.reshape(batchsize,7,7,1024)
        support_tokens = torch.cat((attn_out1,attn_out2),dim=2)
        query_tokens = torch.cat((query_features_img,query_features_mask),dim=2)
        canvas_tokens = torch.cat((support_tokens,query_tokens),dim=1).reshape(batchsize,196,1024)
        return canvas_tokens
    