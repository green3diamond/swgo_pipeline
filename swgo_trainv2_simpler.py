import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Assuming these modules are in the same directory or accessible
from modules import TimestepEmbedder, ResNetBlock, DenseNetwork
from conditional_flow_matching import TargetConditionalFlowMatcher
from dpm import DPM_Solver, NoiseScheduleFlow

# ==============================
#   1. LightningDataModule
# ==============================
# Encapsulates all data loading and processing steps.

class SWGODataModule(pl.LightningDataModule):
    def __init__(self, inputs_path, labels_path, batch_size=512):
        super().__init__()
        self.inputs_path = inputs_path
        self.labels_path = labels_path
        self.batch_size = batch_size
        self.save_hyperparameters() # Saves args to self.hparams

        # Placeholders for normalization stats
        self.x_mean, self.x_std = None, None
        self.c_mean, self.c_std = None, None

    def _normalize_std(self, x, name=""):
        mean = x.mean()
        if x.numel() < 2:
            std = torch.tensor(0.1).float() if name in ["eta", "phi"] else torch.tensor(1.0).float()
        else:
            std = x.std()
            if std == 0:
                std = torch.tensor(0.1).float() if name in ["eta", "phi"] else torch.tensor(1.0).float()
        return (x - mean) / std, mean, std

    def setup(self, stage=None):
        """Called on every GPU."""
        # Load data
        inputs = torch.load(self.inputs_path)
        labels = torch.load(self.labels_path)

        # Reshape and normalize
        Nevents, Nunits, Features = inputs.shape
        inputs_flat = inputs.reshape(Nevents, Nunits * Features)

        x_tensor_norm, self.x_mean, self.x_std = self._normalize_std(labels, name="x")
        c_tensor_norm, self.c_mean, self.c_std = self._normalize_std(inputs_flat, name="cond")
        
        # Store dimensions for model instantiation
        self.x_dim = x_tensor_norm.shape[-1]
        self.c_dim = c_tensor_norm.shape[-1]

        # Create dataset and splits
        dataset = TensorDataset(c_tensor_norm, x_tensor_norm)
        train_size = int(0.8 * len(dataset))
        val_size = int(0.1 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        self.train_set, self.val_set, self.test_set = torch.utils.data.random_split(
            dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, num_workers=4)

# ==============================
#   2. Model Architecture
# ==============================
# This remains the same as it's a standard nn.Module.

class CFMDiagramModel(nn.Module):
    def __init__(self, data_dim, cond_feature_dim, model_dim=128, num_res_blocks=3):
        super().__init__()
        self.time_embedder = TimestepEmbedder(model_dim)
        self.cond_mlp = DenseNetwork(inpt_dim=cond_feature_dim, outp_dim=model_dim, hddn_dim=[2 * model_dim], act_h="silu")
        self.noisy_target_mlp = DenseNetwork(inpt_dim=data_dim, outp_dim=model_dim, hddn_dim=[2 * model_dim], act_h="silu")
        self.conditioning_C = DenseNetwork(inpt_dim=model_dim * 2, outp_dim=model_dim, hddn_dim=[model_dim], act_h="silu")
        self.adaLN_generator = nn.Linear(model_dim, model_dim * 2)
        self.layer_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        nn.init.zeros_(self.adaLN_generator.weight)
        nn.init.zeros_(self.adaLN_generator.bias)
        self.resnet_blocks = nn.ModuleList([ResNetBlock(model_dim, dropout=0.0) for _ in range(num_res_blocks)])
        self.final_layer = DenseNetwork(inpt_dim=model_dim, outp_dim=data_dim, hddn_dim=[model_dim], act_h="silu", ctxt_dim=model_dim)

    def forward(self, x_t, cond, t):
        time_emb = self.time_embedder(t)
        cond_emb = self.cond_mlp(cond)
        combined = torch.cat([time_emb, cond_emb], dim=-1)
        c = self.conditioning_C(combined)
        h = self.noisy_target_mlp(x_t)
        adaLN_params = self.adaLN_generator(c)
        scale, shift = adaLN_params.chunk(2, dim=-1)
        h = self.layer_norm(h) * (1 + scale) + shift
        for block in self.resnet_blocks:
            h = block(h)
        return self.final_layer(h, c)

# ==============================
#   3. LightningModule
# ==============================
# Encapsulates the model, optimizer, and the training/validation logic.

class CFMLightningModule(pl.LightningModule):
    def __init__(self, data_dim, cond_dim, num_res_blocks=6, lr=3e-5):
        super().__init__()
        self.save_hyperparameters() # Saves args to self.hparams
        self.model = CFMDiagramModel(
            data_dim=data_dim,
            cond_feature_dim=cond_dim,
            num_res_blocks=num_res_blocks
        )
        self.criterion = nn.MSELoss()
        self.flow_matcher = TargetConditionalFlowMatcher(sigma=1e-6)

    def forward(self, x_t, cond, t):
        return self.model(x_t, cond, t)

    def _common_step(self, batch, batch_idx):
        cond, x1 = batch
        x0 = torch.randn_like(x1)
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x1)
        v_pred = self(xt, cond, t)
        
        # Calculate loss
        loss = self.criterion(v_pred, ut)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, min_lr=1e-5),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1,
        }
        return [optimizer], [scheduler]

# ==============================
#   4. Training & Evaluation
# ==============================

def main():
    # --- Configuration ---
    INPUTS_PATH = "/n/home04/hhanif/swgo_input_files/mini_inputs.pt"
    LABELS_PATH = "/n/home04/hhanif/swgo_input_files/mini_labels.pt"
    CHECKPOINT_DIR = "/n/home04/hhanif/swgo_testin/"
    BATCH_SIZE = 512
    NUM_EPOCHS = 10
    PATIENCE = 5
    
    # --- Data Setup ---
    data_module = SWGODataModule(inputs_path=INPUTS_PATH, labels_path=LABELS_PATH, batch_size=BATCH_SIZE)
    # Important: Call setup to determine data dimensions
    data_module.setup() 
    
    # --- Model Setup ---
    model = CFMLightningModule(
        data_dim=data_module.x_dim,
        cond_dim=data_module.c_dim,
        num_res_blocks=6,
        lr=3e-5
    )

    # --- Callbacks ---
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename='best_model-{epoch:02d}-{val_loss:.4f}',
        save_top_k=1,
        verbose=True,
        monitor='val_loss',
        mode='min'
    )
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        verbose=True,
        mode='min'
    )

    # --- Trainer ---
    # To run on multiple GPUs, set devices=-1 and strategy='ddp'
    trainer = pl.Trainer(
        accelerator='auto', # 'gpu', 'cpu', etc.
        devices=-1,         # Use all available GPUs
        strategy='ddp',     # Distributed Data Parallel
        max_epochs=NUM_EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback],
        log_every_n_steps=10
    )

    # --- Start Training ---
    print("🚀 Starting training...")
    trainer.fit(model, datamodule=data_module)
    print("✅ Training finished.")

    # --- Evaluation on Test Set ---
    print("🧪 Starting evaluation on the test set...")
    # Load the best model checkpoint
    best_model_path = checkpoint_callback.best_model_path
    print(f"Loading best model from: {best_model_path}")
    trained_model = CFMLightningModule.load_from_checkpoint(best_model_path)
    trained_model.to('cuda' if torch.cuda.is_available() else 'cpu')
    trained_model.eval()

    # Create DPM Solver for sampling
    def model_fn(x, timestep, cond, mask=None, global_data=None):
        # Note: we use trained_model.model to access the nn.Module inside the LightningModule
        return trained_model.model(x, cond, timestep)

    dpm_solver = DPM_Solver(
        model_fn,
        noise_schedule=NoiseScheduleFlow(),
    )

    all_preds, all_truth = [], []
    device = trained_model.device
    test_loader = data_module.test_dataloader()

    with torch.no_grad():
        for cond_batch, x_true_batch in tqdm(test_loader, desc="Generating predictions"):
            cond_batch, x_true_batch = cond_batch.to(device), x_true_batch.to(device)
            x0 = torch.randn_like(x_true_batch)

            x_pred_norm = dpm_solver.sample(
                x0, truth=cond_batch, steps=25, method="multistep", skip_type="time_uniform_flow", mask=None, global_data=None
            )

            # Inverse transform to original scale using stats from the data module
            x_true = x_true_batch * data_module.x_std.to(device) + data_module.x_mean.to(device)
            x_pred = x_pred_norm * data_module.x_std.to(device) + data_module.x_mean.to(device)

            all_truth.append(x_true.cpu().numpy())
            all_preds.append(x_pred.cpu().numpy())

    all_truth = np.concatenate(all_truth, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    
    # --- Plotting Results ---
    print("📊 Generating plots...")
    features = ["x", "y", "E", "theta", "phi"]
    
    # Scatter plots
    plt.figure(figsize=(15, 8))
    for i, f in enumerate(features):
        plt.subplot(2, 3, i+1)
        plt.scatter(all_truth[:, i], all_preds[:, i], alpha=0.3, s=5)
        plt.plot([all_truth[:, i].min(), all_truth[:, i].max()], [all_truth[:, i].min(), all_truth[:, i].max()], "r--")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.title(f"{f}: True vs Pred")
    plt.tight_layout()
    plt.savefig("true_vs_pred_lightning.png", dpi=300)
    plt.close()

    # Distribution plots
    plt.figure(figsize=(15, 8))
    for i, f in enumerate(features):
        plt.subplot(2, 3, i+1)
        plt.hist(all_truth[:, i], bins=50, alpha=0.5, label="True", density=True)
        plt.hist(all_preds[:, i], bins=50, alpha=0.5, label="Predicted", density=True)
        plt.xlabel(f)
        plt.ylabel("Density")
        plt.title(f"{f}: Distribution True vs Pred")
        plt.legend()
    plt.tight_layout()
    plt.savefig("true_vs_pred_distribution_lightning.png", dpi=300)
    plt.close()
    
    print("🎉 All done!")

if __name__ == "__main__":
    main()