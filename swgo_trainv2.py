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
from torch.nn import MultiheadAttention
import os

from conditional_flow_matching import TargetConditionalFlowMatcher
from modules import TimestepEmbedder, ResNetBlock

from dpm import DPM_Solver, NoiseScheduleFlow
from pytorch_lightning.utilities.rank_zero import rank_zero_info

from set2setloss import Set2SetLoss


class MSEAndDirectionLoss(torch.nn.Module):
    """
    Figure 7 - https://arxiv.org/abs/2410.10356
    """

    def __init__(self, cosine_sim_dim: int = 1):  # fix default to 1
        super().__init__()
        self.cosine_sim_dim = cosine_sim_dim

    def forward(self, pred, target, **kwargs):
        mse_loss = torch.nn.functional.mse_loss(pred, target, reduction="sum")
        direction_loss = (
            1.0 - torch.nn.functional.cosine_similarity(pred, target, dim=self.cosine_sim_dim)
        ).sum()
        return mse_loss + direction_loss



class CFMDiagramModel(nn.Module):
    """
    data_dim: number of target features (x0, y0, E, theta, phi) -> 5
    cond_feature_dim: number of conditioned features (x, y, N, T) -> 4
    """
    def __init__(self, data_dim=5, cond_feature_dim=4, model_dim=128,
                 num_res_blocks=3, dropout=0.1, num_detectors=90):
        super().__init__()

        self.num_detectors = num_detectors
        self.model_dim = model_dim

        from modules import TimestepEmbedder, ResNetBlock
        self.time_embedder = TimestepEmbedder(model_dim)

        def two_layer_mlp(in_dim, out_dim, p=0.1):
            h1, h2 = 256, 256
            return nn.Sequential(
                nn.Linear(in_dim, h1), nn.SiLU(), nn.Dropout(p),
                nn.Linear(h1, h2), nn.SiLU(), nn.Dropout(p),
                nn.Linear(h2, out_dim), nn.Dropout(p),
            )

        # Encoders
        self.cond_mlp         = two_layer_mlp(cond_feature_dim, model_dim, p=dropout)
        self.noisy_target_mlp = two_layer_mlp(data_dim,          model_dim, p=dropout)

        # Conditioning signal 'C' (concat time + flattened cond)
        cond_hidden_dim = model_dim
        conditioning_input_dim = (num_detectors * cond_hidden_dim) + model_dim
        self.conditioning_C = two_layer_mlp(conditioning_input_dim, model_dim, p=dropout)

        # Cross-attention
        self.cross_attn = MultiheadAttention(
            embed_dim=model_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True,
        )

        # adaLN-zero
        self.adaLN_generator = nn.Linear(model_dim, model_dim * 2)
        self.layer_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        nn.init.zeros_(self.adaLN_generator.weight)
        nn.init.zeros_(self.adaLN_generator.bias)

        self.resnet_blocks = nn.ModuleList(
            [ResNetBlock(model_dim, dropout=dropout) for _ in range(num_res_blocks)]
        )

        self.final_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim), nn.Dropout(dropout),
        )
        self.output_layer = nn.Linear(model_dim, data_dim)

    def forward(self, x_t: torch.Tensor, cond: torch.Tensor, t: torch.Tensor):
        # Time embedding
        time_emb = self.time_embedder(t)

        # Condition features
        cond_emb = self.cond_mlp(cond)   # (B, M, D) if detectors present
        if cond_emb.dim() == 3:
            pooled_cond = cond_emb.reshape(cond_emb.size(0), -1)  # flatten detectors
        elif cond_emb.dim() == 2:
            pooled_cond = cond_emb
        else:
            raise ValueError(f"Unexpected cond_emb shape {cond_emb.shape}")

        # === Global concatenation context ===
        combined = torch.cat([time_emb, pooled_cond], dim=-1)
        c = self.conditioning_C(combined)

        # Noisy target features
        h = self.noisy_target_mlp(x_t)

        # === Cross-attention ===
        query = h.unsqueeze(1)  # (B, 1, D)
        if cond_emb.dim() == 3:
            key_value = cond_emb   # (B, M, D)
        else:
            key_value = c.unsqueeze(1)  # fallback

        attn_output, _ = self.cross_attn(query, key_value, key_value)
        h = h + attn_output.squeeze(1)  # residual connection

        # === AdaLN conditioning ===
        adaLN_params = self.adaLN_generator(c)
        scale, shift = adaLN_params.chunk(2, dim=-1)
        h = self.layer_norm(h) * (1 + scale) + shift

        for block in self.resnet_blocks:
            h = block(h)

        h = self.final_mlp(h)
        v_pred = self.output_layer(h)
        return v_pred



class SWGODataModule(pl.LightningDataModule):
    """
    Loads directly from .h5, applies normalization, and builds datasets.
    """
    def __init__(self, h5_path="/n/home04/hhanif/swgo_input_files/mini_dataset.h5",
                 batch_size=256, val_split=0.1, seed=42):
        super().__init__()
        self.h5_path = h5_path
        self.batch_size = batch_size
        self.val_split = val_split
        self.seed = seed
        self.num_workers = 10

    def _zscore(self, data: np.ndarray, name: str):
        mean = data.mean()
        std = data.std()
        if std == 0:
            std = 0.1 if name in ["phi"] else 1.0
        return (data - mean) / std

    def setup(self, stage=None):
        import h5py, numpy as np, torch

        # Load from .h5
        with h5py.File(self.h5_path, "r") as f:
            inputs = f["inputs"][:]   # (events, nunits, 4)
            labels = f["labels"][:]   # (events, 5)

        # Normalize inputs
        inputs_norm = np.zeros_like(inputs)
        names_inputs = ["x", "y", "N", "T"]
        for i, name in enumerate(names_inputs):
            arr = inputs[:, :, i]
            if name in ["E", "N"]:  # log-transform for E & N
                arr = np.log(arr + 1e-8)
            inputs_norm[:, :, i] = self._zscore(arr, name)

        # Normalize labels
        labels_norm = np.zeros_like(labels)
        names_labels = ["x0", "y0", "E", "theta", "phi"]
        for i, name in enumerate(names_labels):
            arr = labels[:, i]
            if name in ["E", "N"]:
                arr = np.log(arr + 1e-8)
            labels_norm[:, i] = self._zscore(arr, name)

        # Convert to tensors
        inputs_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_norm, dtype=torch.float32)

        # Build dataset
        full_dataset = TensorDataset(inputs_tensor, labels_tensor)
        n_total = len(full_dataset)
        n_val = int(self.val_split * n_total)
        n_train = n_total - n_val 

        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(self.seed)
        )
        rank_zero_info(f"Data loaded: {n_train} train, {n_val} val")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size,
                          shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size * 2,
                          num_workers=self.num_workers, pin_memory=True)




class CFMLightningModule(pl.LightningModule):
    def __init__(self, learning_rate=3e-5, num_res_blocks=3, flow_matcher_type="target",
                 dropout=0.1, optimizer="adamw", use_scheduler=False, loss_function="mse", time_pow=False, component_weights=None):
        super().__init__()
        self.save_hyperparameters()
        self.model = CFMDiagramModel(num_res_blocks=num_res_blocks, dropout=dropout)

        # Initialize the chosen flow matcher
        if self.hparams.flow_matcher_type == "target":
            self.flow_matcher = TargetConditionalFlowMatcher(sigma=1e-6)
            rank_zero_info("Using TargetConditionalFlowMatcher.")
        elif self.hparams.flow_matcher_type == "reverse":
            self.flow_matcher = ReverseConditionalFlowMatcher(sigma=1e-6)
            rank_zero_info("Using ReverseConditionalFlowMatcher.")
        else:
            raise ValueError(f"Unknown flow_matcher_type: {self.hparams.flow_matcher_type}")

        # Initialize the chosen loss function
        if self.hparams.loss_function == "mse":
            self.criterion = nn.MSELoss()
            rank_zero_info("Using Mean Squared Error (MSE) loss.")
        elif self.hparams.loss_function == "mseanddirection":
            self.criterion = MSEAndDirectionLoss()
        elif self.hparams.loss_function == "component_mse":
            self.criterion = None
            if self.hparams.component_weights is None:
                self.register_buffer("comp_w", torch.ones(5) / 5.0)
            else:
                cw = torch.tensor(self.hparams.component_weights, dtype=torch.float32)
                cw = cw / cw.sum()
                self.register_buffer("comp_w", cw)
            rank_zero_info(f"Using component-wise MSE with weights: {self.comp_w.tolist()}")

        self.set2set_loss = Set2SetLoss()  # uses MSE+Hungarian matching under the hood. :contentReference[oaicite:3]{index=3}


        # Wrap the model into a function usable by DPM_Solver
        def model_fn(x, timestep, cond, mask=None, global_data=None):
            # timestep comes in as (B,) so reshape for broadcasting
            return (1 - timestep.view(-1, 1)) * self.model(x, cond, timestep) + x

        # DPM solver for sampling
        self.dpm = DPM_Solver(model_fn=model_fn, noise_schedule=NoiseScheduleFlow())

    def wasserstein_loss(self, v_pred, v_true):
        """
        Calculates the 1D Wasserstein distance component-wise.
        This is equivalent to the L1 norm (Mean Absolute Error) between the
        sorted predictions and targets over the batch dimension.
        """
        # Sort along the batch dimension (dim=0) for each feature
        v_pred_sorted, _ = torch.sort(v_pred, dim=0)
        v_true_sorted, _ = torch.sort(v_true, dim=0)
        # Return the mean absolute error between the sorted tensors
        return torch.mean(torch.abs(v_pred_sorted - v_true_sorted))

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

        # Loss
        if self.hparams.loss_function == "component_mse":
            per_dim = torch.mean((v_pred - ut) ** 2, dim=0)
            comp_names = ["x0", "y0", "E", "theta", "phi"]
            for i, name in enumerate(comp_names):
                self.log(f"train/{name}_mse", per_dim[i].detach(), on_step=True, on_epoch=True, prog_bar=False, logger=True)
            loss = torch.sum(per_dim * self.comp_w)
        else:
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

        # === Add: prediction sampling and set2set validation loss ===
        cond, x1 = batch                     # x1 is the true target (x0,y0,E,theta,phi) 5-D. :contentReference[oaicite:6]{index=6}
        x1 = x1.view(x1.size(0), -1).to(self.device)

        # sample a predicted target using your existing DPM pipeline (same as predict_step)  :contentReference[oaicite:7]{index=7}
        x0 = torch.randn(cond.size(0), 5, device=cond.device)
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
        x0 = torch.randn(cond.size(0), 5, device=cond.device)
        samples = self.dpm.sample(
            x0,
            truth=cond,
            mask=None,
            global_data=None,
            steps=25,
            method="multistep",
            skip_type="time_uniform_flow",
        )
        return samples

    # === Add: helper to compute pred_loss with Set2SetLoss ===
    @torch.no_grad()
    def get_pred_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        if pred.dim() == 2:
            pred = pred.unsqueeze(1)
        if true.dim() == 2:
            true = true.unsqueeze(1)
        out = self.set2set_loss(pred, true)   # no mask
        return out["total_loss"].to(pred.device)

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

theta_max = np.pi * 65 / 180



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


    parser.add_argument("--input_data_path", type=str, default="/n/home04/hhanif/swgo_input_files/mini_inputs.pt", help="Path to the training dataset (.pt file).")
    parser.add_argument("--label_data_path", type=str, default="/n/home04/hhanif/swgo_input_files/mini_labels.pt", help="Path to the validation dataset (.pt file).")
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size for training.")


    parser.add_argument("--flow_matcher", type=str, default="target", choices=["target", "reverse"], help="Which Conditional Flow Matcher to use.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate used across MLPs and residual blocks.")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["lion", "adamw"], help="Optimizer to use.")
    parser.add_argument("--loss_function", type=str, default="mse", choices=["mse", "mseanddirection", "component_mse"], help="Loss function to use for training.")
    parser.add_argument("--time_pow", action="store_true",
                        help="Use power-law time sampling (t ~ Power(3)), inspired by fs_npf_lightning.")
    parser.add_argument("--use_scheduler", action="store_true", help="Enable the learning rate scheduler.")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated list of GPU IDs to use (e.g., '0,1,2,3').")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for.")
    parser.add_argument("--component_weights", type=float, nargs=5, default=None,
                        help="Optional 5 weights for component_mse over [x0, y0, E, theta, phi]; will be normalized.")
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
    NUM_RES_BLOCKS = 6

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
    
                             
    data_module = SWGODataModule(
        h5_path="/content/drive/MyDrive/Colab Notebooks/SWGO/dataset.h5",
        batch_size=args.batch_size,
        val_split=0.1,
    )

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
    ]

    if args.use_swa:
        callbacks.append(StochasticWeightAveraging(swa_lrs=float(LEARNING_RATE)))
        if is_main_process:
            print("INFO: Stochastic Weight Averaging (SWA) is enabled.")

    if args.save_ckpt:
        # Note: 'train_loss_epoch' is available because self.log in training_step has on_epoch=True
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(output_dir, "ckpts"),
            filename='{epoch:03d}',
            save_top_k=-1,     # keep all checkpoints
            every_n_epochs=1,  # save every epoch
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
