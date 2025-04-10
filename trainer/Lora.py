import torch
from torch import nn

class LoRAParametrization(nn.Module):
    def __init__(self, features_in, features_out, rank=1, alpha=1, device='cpu'):
        super().__init__()
        # Section 4.1 of the paper: 
        #   We use a random Gaussian initialization for A and zero for B, so ∆W = BA is zero at the beginning of training
        self.lora_A = nn.Parameter(torch.zeros((rank,features_out)).to(device))
        self.lora_B = nn.Parameter(torch.zeros((features_in, rank)).to(device))
        nn.init.normal_(self.lora_A, mean=0, std=1)
        
        # Section 4.1 of the paper: 
        #   We then scale ∆Wx by α/r , where α is a constant in r. 
        #   When optimizing with Adam, tuning α is roughly the same as tuning the learning rate if we scale the initialization appropriately. 
        #   As a result, we simply set α to the first r we try and do not tune it. 
        #   This scaling helps to reduce the need to retune hyperparameters when we vary r.
        self.scale = alpha / rank
        self.enabled = True

    def forward(self, original_weights):
        if self.enabled:
            # Return W + (B*A)*scale
            return original_weights + torch.matmul(self.lora_B, self.lora_A).view(original_weights.shape) * self.scale
        else:
            return original_weights


import torch.nn.utils.parametrize as parametrize

def linear_layer_parameterization(layer, device, rank=128, lora_alpha=0.5):
    # Only add the parameterization to the weight matrix, ignore the Bias

    # From section 4.2 of the paper:
    #   We limit our study to only adapting the attention weights for downstream tasks and freeze the MLP modules (so they are not trained in downstream tasks) both for simplicity and parameter-efficiency.
    #   [...]
    #   We leave the empirical investigation of [...], and biases to a future work.
    
    features_in, features_out = layer.weight.shape
    return LoRAParametrization(
        features_in, features_out, rank=rank, alpha=lora_alpha, device=device
    )


def load_lora_state_dict(model, lora_state_dict):
    for name, module in model.named_modules():
        if hasattr(module, 'parametrizations') and 'weight' in module.parametrizations:
            lora_params = module.parametrizations['weight'][0]
            if f'{name}.lora_A' in lora_state_dict:
                lora_params.lora_A.data = lora_state_dict[f'{name}.lora_A']
            if f'{name}.lora_B' in lora_state_dict:
                lora_params.lora_B.data = lora_state_dict[f'{name}.lora_B']
    
def save_lora_state_dict(model):
    lora_state_dict = {}

    for name, module in model.named_modules():
        if hasattr(module, 'parametrizations') and 'weight' in module.parametrizations:
            lora_params = module.parametrizations['weight'][0] 
            lora_state_dict[f'{name}.lora_A'] = lora_params.lora_A
            lora_state_dict[f'{name}.lora_B'] = lora_params.lora_B
    return lora_state_dict

def freeze_base_weights(model):
    for name, module in model.named_modules():
        if hasattr(module, 'parametrizations') and 'weight' in module.parametrizations:
            if hasattr(module.parametrizations['weight'][0], 'lora_A'):
                module.parametrizations['weight'][0].lora_A.requires_grad = True
                module.parametrizations['weight'][0].lora_B.requires_grad = True
            else:
                pass
