import torch
torch.set_float32_matmul_precision("medium")
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader, random_split
import math
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.loggers import CometLogger
import os
from datetime import datetime
import argparse
from lion_opt import Lion
from conditional_flow_matching import TargetConditionalFlowMatcher
from custom_scheduler import CosineAnnealingWarmupRestarts
from modules import TimestepEmbedder, ResNetBlock

from dpm import DPM_Solver, NoiseScheduleFlow
from pytorch_lightning.utilities.rank_zero import rank_zero_info
from modules import SWGODataModule

def normalize_std(x, name=""):
    mean = x.mean()
    if x.numel() < 2:
        std = torch.tensor(0.1).float() if name in ["eta", "phi"] else torch.tensor(1.0).float()
    else:
        std = x.std()
        if std == 0:
            std = torch.tensor(0.1).float() if name in ["eta", "phi"] else torch.tensor(1.0).float()
    return (x - mean) / std, mean, std


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



inputs_path = "/n/home04/hhanif/swgo_input_files/mini_inputs.pt"
labels_path = "/n/home04/hhanif/swgo_input_files/mini_labels.pt"

inputs = torch.load(inputs_path).to(device)
labels = torch.load(labels_path).to(device)

Nevents, Nunits, Features = inputs.shape
inputs_flat = inputs.reshape(Nevents, Nunits * Features)

x_tensor_norm, x_mean, x_std = normalize_std(labels, name="x")
c_tensor_norm, c_mean, c_std = normalize_std(inputs_flat, name="cond")

dataset = TensorDataset(c_tensor_norm, x_tensor_norm)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=512, shuffle=True)
val_loader = DataLoader(val_set, batch_size=512)
test_loader = DataLoader(test_set, batch_size=512)


x_dim = x_tensor_norm.shape[-1]
c_dim = c_tensor_norm.shape[-1]

class CFMDiagramModel(nn.Module):
    """
    data_dim: number of target features (x0, y0, E, theta, phi) -> 5
    cond_feature_dim: number of conditioned features (x, y, N, T) -> 4
    """
    def __init__(self, data_dim=x_dim, cond_feature_dim=c_dim, model_dim=128, num_res_blocks=3, dropout=0.1):
        super().__init__()

        self.time_embedder = TimestepEmbedder(model_dim)

        def two_layer_mlp(in_dim, out_dim, hidden_dim=None, p=0.1):
            h = hidden_dim or out_dim
            return nn.Sequential(
                nn.Linear(in_dim, h),
                nn.SiLU(),
                nn.Dropout(p),
                nn.Linear(h, out_dim),
                nn.Dropout(p),
            )

        # Encoders
        self.cond_mlp         = two_layer_mlp(cond_feature_dim, model_dim, p=dropout)
        self.noisy_target_mlp = two_layer_mlp(data_dim,          model_dim, p=dropout)

        # Conditioning signal 'C' (concat time + pooled cond)
        self.conditioning_C = two_layer_mlp(model_dim * 2, model_dim, p=dropout)

        # adaLN-zero
        self.adaLN_generator = nn.Linear(model_dim, model_dim * 2)
        self.layer_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        nn.init.zeros_(self.adaLN_generator.weight)
        nn.init.zeros_(self.adaLN_generator.bias)

        # Residual stack
        self.resnet_blocks = nn.ModuleList([ResNetBlock(model_dim, dropout=dropout) for _ in range(num_res_blocks)])

        # Final projection
        self.final_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
            nn.Dropout(dropout),
        )
        self.output_layer = nn.Linear(model_dim, data_dim)


    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor):
        # Time embedding
        time_emb = self.time_embedder(t)

        cond_emb = self.cond_mlp(cond)
        pooled_cond = cond_emb
    # Concatenate for context 'C'
        combined = torch.cat([time_emb, pooled_cond], dim=-1)
        c = self.conditioning_C(combined)

        # Noisy target features
        h = self.noisy_target_mlp(x_t)

        # adaLN-zero
        adaLN_params = self.adaLN_generator(c)
        scale, shift = adaLN_params.chunk(2, dim=-1)
        h = self.layer_norm(h) * (1 + scale) + shift

        # Residual blocks
        for block in self.resnet_blocks:
            h = block(h)

        # Final MLP
        h = self.final_mlp(h)
        v_pred = self.output_layer(h)
        return v_pred

class CFMLightningModule(pl.LightningModule):
    def __init__(self, learning_rate=3e-5, num_res_blocks=3, flow_matcher_type="target",
                 dropout=0.1, optimizer="adamw", use_scheduler=False, loss_function="mse", time_pow=False, component_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = CFMDiagramModel(num_res_blocks=num_res_blocks, dropout=dropout)

        # Initialize the chosen flow matcher
        self.flow_matcher = TargetConditionalFlowMatcher(sigma=1e-6)
        rank_zero_info("Using TargetConditionalFlowMatcher.")


        self.criterion = nn.MSELoss()
        rank_zero_info("Using Mean Squared Error (MSE) loss.")

        # Wrap the model into a function usable by DPM_Solver
        def model_fn(x, timestep, cond, mask=None, global_data=None):
            # timestep comes in as (B,) so reshape for broadcasting
            return (1 - timestep.view(-1, 1)) * self.model(x, cond, timestep) + x

        # DPM solver for sampling
        self.dpm = DPM_Solver(model_fn=model_fn, noise_schedule=NoiseScheduleFlow())



    def forward(self, x_t, cond, t):
        return self.model(x_t, cond, t)

    def _common_step(self, batch, batch_idx):
        # batch = (cond, x1)
        cond, x1 = batch
        x1 = x1.view(x1.size(0), -1)         # ensure (B, 5)
        x0 = torch.randn_like(x1)            # prior

        # Sample time, noisy data xt, and the conditional vector field ut
        t_override = None
        if getattr(self.hparams, "time_pow", False):
            t_override = torch.tensor(np.random.power(3, size=x1.shape[0]), dtype=x1.dtype, device=x1.device)
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x1, t=t_override)

        # Predict vector field
        v_pred = self(xt, cond, t)

        loss = self.criterion(v_pred, ut)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Compute loss and also log per-dimension terms if requested
        cond, x1 = batch
        x1 = x1.view(x1.size(0), -1).to(self.device)
        x0 = torch.randn_like(x1)
        t_override = None
        if getattr(self.hparams, "time_pow", False):
            t_override = torch.tensor(np.random.power(3, size=x1.shape[0]), dtype=x1.dtype, device=x1.device)
        t, xt, ut = self.flow_matcher.sample_location_and_conditional_flow(x0, x1, t=t_override)
        v_pred = self(xt, cond, t)
        if self.hparams.loss_function == "component_mse":
            per_dim = torch.mean((v_pred - ut) ** 2, dim=0)
            comp_names = ["x0", "y0", "E", "theta", "phi"]
            for i, name in enumerate(comp_names):
                self.log(f"val/{name}_mse", per_dim[i].detach(), on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            loss = torch.sum(per_dim * self.comp_w)
        else:
            loss = self.criterion(v_pred, ut)

        cond, x1 = batch                     # x1 is the true target (x0,y0,E,theta,phi) 5-D. :contentReference[oaicite:6]{index=6}
        x1 = x1.view(x1.size(0), -1).to(self.device)

        # sample a predicted target using your existing DPM pipeline (same as predict_step)  :contentReference[oaicite:7]{index=7}
        x0 = torch.randn(cond.size(0), 5, device=self.device)
        pred = self.dpm.sample(
            x0,
            truth=cond,
            mask=None,
            global_data=None,
            steps=50,
            method="multistep",
            skip_type="time_uniform_flow",
            show_progress=False,   # <--- silences tqdm

        )

        pred_loss = self.get_pred_loss(pred, x1)

        # Log both
        self.log("val_loss",      loss,      on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log("val_pred_loss", pred_loss, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        # === End Add ===

        return {"val_loss": loss, "val_pred_loss": pred_loss}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        cond, _ = batch  # We only need the condition from the batch
        x0 = torch.randn(cond.size(0), 5, device=self.device)
        samples = self.dpm.sample(
            x0,
            truth=cond,
            mask=None,
            global_data=None,
            steps=50,
            method="multistep",
            skip_type="time_uniform_flow",
        )
        return samples



        # Build a mask of shape (B, N_pred, 2) where the second column is 1s,
        # matching what Set2SetLoss.forward expects (it uses mask[..., 1]). :contentReference[oaicite:5]{index=5}
        mask = torch.zeros((B, N_pred, 2), device=device, dtype=pred.dtype)
        mask[..., 1] = 1.0

        # out: dict with "total_loss", "pt_loss", "eta_loss", "phi_loss" fields. We use total_loss.
        return out["total_loss"].to(device)

    def configure_optimizers(self):
        base_lr = float(self.hparams.learning_rate)

        if self.hparams.optimizer == "lion":
            rank_zero_info(f"Using Lion optimizer with learning_rate={base_lr}")
            optimizer = Lion(self.parameters(), lr=base_lr)
        elif self.hparams.optimizer == "adamw":
            rank_zero_info(f"Using AdamW optimizer with learning_rate={base_lr}")
            optimizer = torch.optim.AdamW(self.parameters(), lr=base_lr)
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer}")

        if not self.hparams.use_scheduler:
            rank_zero_info("Learning rate scheduler is disabled.")
            return optimizer

        rank_zero_info("Using CosineAnnealingWarmupRestarts scheduler.")
        default_scheduler_dict = {
            "first_cycle_steps": 10,
            "warmup_steps": 4,
            "max_lr": 4 * base_lr,
            "min_lr": max(1e-6, base_lr / 3),
            "gamma": 0.8,
        }
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            **default_scheduler_dict,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}




class LossTracker(pl.Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
    def on_train_epoch_end(self, trainer, pl_module):
        # Access the logged metric which is aggregated over the epoch
        if 'train_loss_epoch' in trainer.logged_metrics:
            self.train_losses.append(trainer.logged_metrics['train_loss_epoch'].item())
    def on_validation_epoch_end(self, trainer, pl_module):
        if 'val_loss' in trainer.logged_metrics:
            self.val_losses.append(trainer.logged_metrics['val_loss'].item())

if __name__ == "__main__":
    # Check if this is the main process (rank 0) in a distributed setup.
    # This is crucial for ensuring that file I/O and logging only happen once.
    is_main_process = os.environ.get("LOCAL_RANK", "0") == "0"
    
    if is_main_process:
        print(f"PyTorch using device: {device}")

    parser = argparse.ArgumentParser(description="Train a Conditional Flow Matching model.")


    parser.add_argument("--use_comet", action="store_true", help="Enable Comet.ml logging.")
    parser.add_argument("--run_name", type=str, default="", help="Optional custom name for the output folder.")


    parser.add_argument("--train_data_path", type=str, default="/n/home04/hhanif/swgo_input_files/train.pt", help="Path to the training dataset (.pt file).")
    parser.add_argument("--val_data_path", type=str, default="/n/home04/hhanif/swgo_input_files/val.pt", help="Path to the validation dataset (.pt file).")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training.")


    parser.add_argument("--flow_matcher", type=str, default="target", choices=["target", "reverse"], help="Which Conditional Flow Matcher to use.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate used across MLPs and residual blocks.")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["lion", "adamw"], help="Optimizer to use.")
    parser.add_argument("--loss_function", type=str, default="mse", choices=["mse", "mseanddirection", "component_mse"], help="Loss function to use for training.")
    parser.add_argument("--time_pow", action="store_true",
                        help="Use power-law time sampling (t ~ Power(3)), inspired by fs_npf_lightning.")
    parser.add_argument("--component_weights", type=float, nargs=5, default=None,
                        help="Optional 5 weights for component_mse over [x0, y0, E, theta, phi]; will be normalized.")
    parser.add_argument("--use_scheduler", action="store_true", help="Enable the learning rate scheduler.")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3').")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for.")

    parser.add_argument("--gradient_clip_val", type=float, default=0.0, help="Value for gradient clipping. 0 means no clipping.")
    parser.add_argument("--use_swa", action="store_true", help="Enable Stochastic Weight Averaging.")

    parser.add_argument("--save_ckpt", action="store_true", help="Enable saving model checkpoints.")

    args = parser.parse_args()

    pl.seed_everything(42)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create dynamic output directory
    base_script_name = os.path.splitext(os.path.basename(__file__))[0]
    if args.run_name:
        output_dir = f"{args.run_name}_{timestamp}"
    else:
        output_dir = f"{base_script_name}_run_results_{timestamp}"
    log_dir = os.path.join(output_dir, "slurm_logs")
    
    if is_main_process:
        os.makedirs(log_dir, exist_ok=True)


    # --- Hyperparameters ---
    EPOCHS      = args.epochs
    LEARNING_RATE = 3e-5
    NUM_RES_BLOCKS = 4

    # --- Comet.ml Logger ---
    logger = None
    if args.use_comet and is_main_process:
        print("Comet.ml logging is enabled.")
        try:
            exp_name = os.path.basename(output_dir)
            logger = CometLogger(
                api_key=os.environ.get("COMET_API_KEY"),
                project_name=os.environ.get("COMET_PROJECT_NAME", "swgo-v2"),
                workspace=os.environ.get("COMET_WORKSPACE"),
                experiment_name=exp_name
            )
            print(f"CometLogger initialized successfully for experiment: {exp_name}")
        except ImportError:
            print("Comet.ml is not installed. Please run 'pip install comet_ml'. Disabling logger.")
        except Exception as e:
            print(f"Error initializing CometLogger: {e}")
            print("Please ensure COMET_API_KEY, COMET_PROJECT_NAME, and COMET_WORKSPACE env vars are set. Disabling logger.")


    data_module = SWGODataModule(train_data_path=args.train_data_path, val_data_path=args.val_data_path,
                                 batch_size=args.batch_size)

    model_module = CFMLightningModule(
        learning_rate=LEARNING_RATE,
        num_res_blocks=NUM_RES_BLOCKS,
        flow_matcher_type=args.flow_matcher,
        dropout=args.dropout,
        optimizer=args.optimizer,
        use_scheduler=args.use_scheduler,
        loss_function=args.loss_function,
        time_pow=args.time_pow,
        component_weights=args.component_weights
    )

    loss_tracker = LossTracker()

    # Configure callbacks: progress bar, loss tracking, early stopping, and optional checkpointing
    callbacks = [
        TQDMProgressBar(refresh_rate=10),
        loss_tracker,
        EarlyStopping(monitor="val_loss", mode="min", patience=25, verbose=is_main_process)
    ]

    if args.use_swa:
        callbacks.append(StochasticWeightAveraging(swa_lrs=float(LEARNING_RATE)))
        if is_main_process:
            print("INFO: Stochastic Weight Averaging (SWA) is enabled.")

    if args.save_ckpt:
        # Note: 'train_loss_epoch' is available because self.log in training_step has on_epoch=True
        checkpoint_callback = ModelCheckpoint(
            dirpath=output_dir,
            filename='{epoch:03d}-{val_loss:.4f}-{train_loss_epoch:.4f}',
            monitor='val_loss',
            mode='min',
            save_top_k=3,
            verbose=is_main_process
        )
        callbacks.append(checkpoint_callback)
        if is_main_process:
            print(f"INFO: Model checkpoints will be saved to '{output_dir}'.")
    else:
        if is_main_process:
            print("INFO: Model checkpointing is disabled. Use --save_ckpt to enable.")

    # Configure trainer for CPU, single GPU, or multi-GPU
    accelerator = "cpu"
    devices = 1
    strategy = "auto"

    if torch.cuda.is_available():
        accelerator = "gpu"
        try:
            gpu_ids = [int(i.strip()) for i in args.gpus.split(',')]
            devices = gpu_ids
            if len(gpu_ids) > 1:
                strategy = "ddp"
                if is_main_process:
                    print(f"INFO: Using DistributedDataParallel strategy on GPUs: {gpu_ids}")
            else:
                if is_main_process:
                    print(f"INFO: Using single GPU: {gpu_ids}")
        except ValueError:
            if is_main_process:
                print(f"ERROR: Invalid --gpus argument '{args.gpus}'. Please use a comma-separated list of integers (e.g., '0,1,2').")
            exit()
    else:
        if is_main_process:
            print("WARNING: CUDA not available. Training on CPU.")

    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        callbacks=callbacks,
        log_every_n_steps=50,
        logger=logger,
        default_root_dir=output_dir,
        gradient_clip_val=args.gradient_clip_val
    )

    if args.use_comet and trainer.logger:
        hyperparams_to_log = {**model_module.hparams, "batch_size": args.batch_size}
        trainer.logger.log_hyperparams(hyperparams_to_log)

    # --- Training ---
    if is_main_process:
        print("\nStarting training with PyTorch Lightning...")
    trainer.fit(model_module, data_module)
    if is_main_process:
        print("Training finished.")


    if is_main_process:
        print("Job finished.")