import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from data_loader import SWGODataModule

from swgo_trainv2 import  LossTracker
from xaims import flows, ddpm, sde, visualize, transforms, aux

# For reproducibility
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

# Use device: GPU if available
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

# Paths
inputs_path = "/n/home04/hhanif/swgo_input_files/mini_inputs.pt"
labels_path = "/n/home04/hhanif/swgo_input_files/mini_labels.pt"

# Load tensors
inputs = torch.load(inputs_path)
labels = torch.load(labels_path)



labels = labels.to(device)   # send labels to GPU if available
inputs = inputs.to(device)   # do the same for inputs if you’ll use them


Nevents, Nunits, Features = inputs.shape

# Flatten detectors into feature dimension
inputs_flat = inputs.reshape(Nevents, Nunits * Features)


zscore_map_x   = transforms.ZScoreTransform(labels.mean(dim=0, keepdim=True),
                                            labels.std(dim=0, keepdim=True)).to(device)
x_tensor_norm  = zscore_map_x.forward(labels)
pre_transform_x  = transforms.CompositeTransform([zscore_map_x]).to(device) # 3. Chain to a total transform map to reverse later in the inference




zscore_map_c   = transforms.ZScoreTransform(inputs_flat.mean(dim=0, keepdim=True),
                                            inputs_flat.std(dim=0, keepdim=True)).to(device)
c_tensor_norm  = zscore_map_c.forward(inputs_flat)

pre_transform_c  = transforms.CompositeTransform([zscore_map_c]).to(device) # 3. Chain to a total transform map to reverse later in the inference



x_dim   = x_tensor_norm.shape[-1]
c_dim   = c_tensor_norm.shape[-1]


from xaims import ddpm, aux, visualize

n_epochs   = 1
batch_size = 256

# Create dataloaders
train_loader, val_loader, test_loader = aux.split_loaders(x_tensor_norm, c_tensor_norm, frac_train=0.8,frac_val=0.1, batch_size=batch_size)

lr              = 1e-3
weight_decay    = 1e-3

time_embed_dim  = 8
model_ddpm = ddpm.DDPM(
    x_dim           = x_dim,
    cond_dim        = c_dim,
    beta_start      = 1e-4,
    beta_end        = 0.02,
    diffusion_steps = 1000,
    time_embed_dim  = 8,
    nnet            = aux.ResidualMLP(in_dim=x_dim + time_embed_dim + c_dim, out_dim=x_dim,
                               **{"hidden_dim": [96, 96, 96], "layer_norm": True, "act": "silu", "dropout": 0})
    #nnet           = aux.VectorTransformer(in_dim=x_dim + time_embed_dim + c_dim, out_dim=x_dim,
    #                           **{"d_model": 32, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "dropout": 0.0})
).to(device)

# Optimizer and scheduler
optimizer_ddpm = optim.AdamW(model_ddpm.parameters(), lr=lr, weight_decay=weight_decay)
scheduler_ddpm = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ddpm, 'min', factor=0.5, patience=10, min_lr=1e-5)

# Returns the best validation loss model
losses_ddpm, model_ddpm = aux.train_wrapper(model=model_ddpm, optimizer=optimizer_ddpm, scheduler=scheduler_ddpm,
                                  train_loader=train_loader, val_loader=val_loader, n_epochs=n_epochs)

fix,axs = visualize.plot_losses(train_losses=losses_ddpm[0], val_losses=losses_ddpm[1] , save_path="ddpm_loss.png")


# DDPM diffusion model sampler

def sample_from_ddpm(num_samples: int, cond: torch.Tensor, device='cpu'):
    
    model_ddpm.eval() #!
    
    with torch.no_grad():
        cond      = aux.expand_dim(cond, num_samples=num_samples, device=device) # Expand batch dimensions
        cond_norm = pre_transform_c.forward(cond) # Forward processing
        
        x         = model_ddpm.sample(num_samples, cond_norm)        
        samples   = pre_transform_x.reverse(x)    # Revert pre-processing
    
    return samples


x_ddpm = sample_from_ddpm(num_events, cond=sqrts, device=device)
