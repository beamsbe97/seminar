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
class PromptGeneratorlimzero(nn.Module):
    def __init__(self,args,dropout = 0):
        super().__init__()
        self.CrossAttention_S = nn.MultiheadAttention(embed_dim = 1024, dropout = dropout,num_heads = 8,batch_first=True)
        self.SelfAttention_Q = nn.MultiheadAttention(embed_dim = 1024, dropout = dropout, num_heads = 8,batch_first=True)
        if args.G_copy_another:
            self.CrossAttention_SM = nn.MultiheadAttention(embed_dim = 1024, dropout = dropout,num_heads = 8,batch_first=True)
        self.SelfAttention_S = nn.MultiheadAttention(embed_dim = 1024, dropout = dropout, num_heads = 8,batch_first=True)
        print('dropout ',dropout)
        print('Zero\n')
        self.Layer_norm_s = nn.LayerNorm(1024)
        self.Layer_norm_q = nn.LayerNorm(1024)
        self.Linear = nn.Linear(1024,1024)
        self.args = args
        self.initialize_weights()

    def initialize_weights(self):
        self.apply(self._init_weights)
        nn.init.eye_(self.Linear.weight)
        nn.init.zeros_(self.Linear.bias)

    def _init_weights(self,m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, support_features, query_features):
        batchsize = support_features.shape[0]
        N = support_features.shape[1] 
        qss = support_features.reshape(batchsize*N,98,1024)
        if self.args.align_s:
            qss_layer_norm = self.Layer_norm_s(qss)
            ats_ans,_ = self.SelfAttention_S(qss_layer_norm,qss_layer_norm,qss_layer_norm)
            support_features = qss + ats_ans #[B*N,98,1024]
        else :
            support_features = qss
        query_features = query_features.reshape(batchsize,1,7,14,1024)
        query_features_img = query_features[:,:,:,:7,:]
        query_features_mask = query_features[:,:,:,7:,:]
        query_features_img = query_features_img.reshape(batchsize,49,1024)
        if self.args.align_q:
            qsq_layner_norm = self.Layer_norm_q(query_features_img)
            atq_ans,_ = self.SelfAttention_Q(qsq_layner_norm,qsq_layner_norm,qsq_layner_norm)
            query_features_img = query_features_img + atq_ans #[B,49,1024]
        query_features_img = query_features_img.reshape(batchsize*49,1,1024)
        support_features = support_features.reshape(batchsize*N,7,14,1024)
        support_features_img = support_features[:,:,:7,:]
        support_features_mask = support_features[:,:,7:,:]
        support_features_img = support_features_img.reshape(batchsize,N,49,1024)
        support_features_mask = support_features_mask.reshape(batchsize,N,49,1024)
        support_features_img = support_features_img.permute(0,2,1,3).reshape(batchsize*49,N,1024)
        support_features_mask = support_features_mask.permute(0,2,1,3).reshape(batchsize*49,N,1024)
        attn_out1,attn_weight = self.CrossAttention_S(query_features_img,support_features_img,support_features_img)         #[B*49,1,1024]
        # --- regularizer metrics: computed EVERY step regardless of lambda, so the
        #     baseline (lambda=0) run records the same telemetry for comparison ---
        support_norm_shared = F.normalize(support_features_img, dim=-1)
        # confidence penalty: attention mass placed on prompts with low query similarity
        query_norm = F.normalize(query_features_img, dim=-1)
        per_prompt_sim = torch.bmm(query_norm, support_norm_shared.transpose(1, 2))
        conf_penalty = (attn_weight * (1 - per_prompt_sim.detach())).sum(dim=(1,2)).mean()
        # diversity: mean pairwise cosine similarity among the N candidate prompts
        N_val = support_features_img.shape[1]
        if N_val > 1:
            cos_sim = torch.bmm(support_norm_shared, support_norm_shared.transpose(1, 2))
            triu_mask = torch.triu(torch.ones(N_val, N_val, dtype=torch.bool, device=cos_sim.device), diagonal=1)
            diversity_loss = cos_sim[:, triu_mask].mean()
        else:
            diversity_loss = torch.zeros((), device=support_features_img.device)
        attn_out2 = (attn_weight @ (self.Linear(support_features_mask)))          #[B*49,1,1024]
        if self.args.G_copy_another:
            attn_out2,attn_weight = self.CrossAttention_SM(query_features_img,support_features_mask,support_features_mask)
        if self.args.G_only_div:
            attn_out1 = support_features_img.mean(dim=1, keepdim=True)
            attn_out2 = support_features_mask.mean(dim=1, keepdim=True)
        attn_out1 = attn_out1.reshape(batchsize,7,7,1024)
        attn_out2 = attn_out2.reshape(batchsize,7,7,1024)
        query_features_img = query_features_img.reshape(batchsize,7,7,1024)
        query_features_mask = query_features_mask.reshape(batchsize,7,7,1024)
        loss = 0
        if self.args.loss_choice == 'cos':
            cosine_loss_img = F.cosine_embedding_loss(query_features_img.reshape(batchsize*49,1024),attn_out1.reshape(batchsize*49,1024), torch.tensor([1]).to(self.args.device))
            cosine_loss_msk = F.cosine_embedding_loss(query_features_mask.reshape(batchsize*49,1024),attn_out2.reshape(batchsize*49,1024), torch.tensor([1]).to(self.args.device))
            loss = cosine_loss_img+cosine_loss_msk
        if self.args.loss_choice == 'l1':
            l1_loss_img = F.l1_loss(query_features_img, attn_out1)
            l1_loss_msk = F.l1_loss(query_features_mask, attn_out2)
            loss = l1_loss_img+l1_loss_msk
        if self.args.loss_choice == 'l2':
            l2_loss_img = F.mse_loss(query_features_img, attn_out1)
            l2_loss_msk = F.mse_loss(query_features_mask, attn_out2)
            loss = l2_loss_img + l2_loss_msk
        support_tokens = torch.cat((attn_out1,attn_out2),dim=2)
        query_tokens = torch.cat((query_features_img,query_features_mask),dim=2)
        canvas_tokens = torch.cat((support_tokens,query_tokens),dim=1).reshape(batchsize,196,1024)
        # raw pre-alignment term (before lamba weighting) captured for telemetry
        l_pa_raw = loss if torch.is_tensor(loss) else torch.zeros((), device=support_features_img.device)
        loss = loss * self.args.lamba
        if self.args.diversity_lambda > 0:
            loss = loss + self.args.diversity_lambda * diversity_loss
        if self.args.conf_lambda > 0:
            loss = loss + self.args.conf_lambda * conf_penalty
        # expose RAW (unweighted) component values so the training loop can log them.
        # weighted contributions are recoverable as lamba*l_pa, diversity_lambda*l_div, conf_lambda*l_conf
        self.loss_terms = {
            'l_pa': l_pa_raw.detach(),
            'l_div': diversity_loss.detach(),
            'l_conf': conf_penalty.detach(),
        }
        return canvas_tokens,loss
