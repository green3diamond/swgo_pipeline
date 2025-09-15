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
from xaims import flows, ddpm, sde, visualize, transforms


# ---------------------------
# Utilities
# ---------------------------


def pool_cond(cond: torch.Tensor) -> torch.Tensor:
    """
    Pools condition tensor by taking the mean over the sequence dimension.
    Handles [B, N, D] -> [B, D] and [N, D] -> [D].
    """
    if cond.dim() == 3:  # Batched input
        return cond.mean(dim=1)
    if cond.dim() == 2:  # Single sample from dataset iteration
        return cond.mean(dim=0)
    return cond

def build_transforms(train_dataset, device="cpu"):
    conds, xs = [], []
    for cond, x in train_dataset:
        cond = cond.squeeze()          # [90,4]
        cond = pool_cond(cond)         # 🔑 pool -> [4]
        x    = x.squeeze()             # [5]

        conds.append(cond.unsqueeze(0).float())
        xs.append(x.unsqueeze(0).float())

    cond_all = torch.cat(conds, dim=0)  # [N,4]
    x_all    = torch.cat(xs, dim=0)     # [N,5]

    # Z-score for x
    zscore_map_x = transforms.ZScoreTransform(x_all.mean(0, keepdim=True),
                                              x_all.std(0, keepdim=True)).to(device)
    pre_transform_x = transforms.CompositeTransform([zscore_map_x]).to(device)

    # Z-score for cond
    zscore_map_c = transforms.ZScoreTransform(cond_all.mean(0, keepdim=True),
                                              cond_all.std(0, keepdim=True)).to(device)
    pre_transform_c = transforms.CompositeTransform([zscore_map_c]).to(device)

    return pre_transform_x, pre_transform_c



def make_loaders(dm: SWGODataModule, batch_size: int, device="cpu"):
    pre_transform_x, pre_transform_c = build_transforms(dm.train_dataset, device)

    def apply_transforms(dataset):
        conds, xs = [], []
        for cond, x in dataset:
            cond = cond.squeeze()      # [90,4]
            cond = pool_cond(cond)     # [4]
            x    = x.squeeze()         # [5]

            conds.append(pre_transform_c.forward(cond.unsqueeze(0)))  # [1,4]
            xs.append(pre_transform_x.forward(x.unsqueeze(0)))        # [1,5]

        return torch.cat(conds, dim=0), torch.cat(xs, dim=0)

    train_cond, train_x = apply_transforms(dm.train_dataset)
    val_cond, val_x     = apply_transforms(dm.val_dataset)

    train_dataset = torch.utils.data.TensorDataset(train_cond, train_x)
    val_dataset   = torch.utils.data.TensorDataset(val_cond, val_x)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader   = torch.utils.data.DataLoader(val_dataset,   batch_size=batch_size*2, shuffle=False, num_workers=1)

    return train_loader, val_loader, pre_transform_x, pre_transform_c



# ---------------------------
# Lightning wrappers
# ---------------------------

class FlowLightningModule(pl.LightningModule):
    def __init__(self, lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = flows.CouplingFlow(
            x_dim=5, cond_dim=4,
            flow_layers=4,
            flow_type="spline",
            permute=True,
            nn_param=[
                {"hidden_dim": [96, 96], "layer_norm": True, "act": "silu", "dropout": 0.01},
                {"hidden_dim": [96, 96], "layer_norm": True, "act": "silu", "dropout": 0.01},
            ],
        )
    def training_step(self, batch, _):
        cond, x = batch
        loss = -self.model.log_prob(x, cond).mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    def validation_step(self, batch, _):
        cond, x = batch
        loss = -self.model.log_prob(x, cond).mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    def configure_optimizers(self):
        opt = optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=20, min_lr=1e-5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}


class DDPM_Lightning(pl.LightningModule):
    def __init__(self, lr=1e-3, weight_decay=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = ddpm.DDPM(x_dim=5, cond_dim=4,
                               beta_start=1e-4, beta_end=0.02, diffusion_steps=1000,
                               time_embed_dim=8, nnet=None)
    def training_step(self, batch, _):
        cond, x = batch
        loss = self.model.loss(x, cond)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True); return loss
    def validation_step(self, batch, _):
        cond, x = batch
        loss = self.model.loss(x, cond)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True); return loss
    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=10, min_lr=1e-5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}
    def sample(self, n, cond): return self.model.sample(n, cond)


class SDE_Lightning(pl.LightningModule):
    def __init__(self, lr=1e-3, weight_decay=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = sde.SDEModel(
            sde=sde.VPSDE(beta_0=0.1, beta_1=20.0),
            x_dim=5, cond_dim=4, time_embed_dim=8,
            loss_weighting=False, nnet=None)
    def training_step(self, batch, _):
        cond, x = batch
        loss = self.model.loss(x, cond)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True); return loss
    def validation_step(self, batch, _):
        cond, x = batch
        loss = self.model.loss(x, cond)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True); return loss
    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=10, min_lr=1e-5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}
    def sample(self, n, cond, steps=1000, use_ode=False): return self.model.sample(n, cond, steps, use_ode)


# ---------------------------
# Training / Evaluation
# ---------------------------

def build_trainer(epochs: int, ckpt_prefix: str, patience: int = 20):
    loss_tracker = LossTracker()
    callbacks = [
        TQDMProgressBar(refresh_rate=10),
        EarlyStopping(monitor="val_loss", mode="min", patience=patience),
        ModelCheckpoint(dirpath=f"checkpoints/{ckpt_prefix}",
                        filename="{epoch:02d}-{val_loss:.4f}",
                        monitor="val_loss", save_top_k=3, mode="min"),
        loss_tracker,
    ]
    trainer = pl.Trainer(max_epochs=epochs,
                         accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         devices=1, callbacks=callbacks, log_every_n_steps=50)
    return trainer, loss_tracker


def plot_predictions(y_true, y_pred, out_path, names=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    D = y_true.shape[1]; cols = min(3, D); rows = int(np.ceil(D / cols))
    plt.figure(figsize=(5*cols, 3.5*rows))
    for d in range(D):
        plt.subplot(rows, cols, d+1)
        plt.hist(y_true[:, d], bins=60, histtype="step", label="True", density=True)
        plt.hist(y_pred[:, d], bins=60, histtype="step", label="Pred", density=True)
        plt.title(names[d] if names and d < len(names) else f"dim {d}")
        plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_path); plt.close()


def run_predictions_and_plots(model_type, model, pre_transform_x, pre_transform_c, test_path, out_dir="results"):
    test_inputs, test_labels = torch.load(test_path)
    cond = pool_cond(test_inputs.squeeze()).float()
    x_true = test_labels.squeeze().float()

    cond_z = pre_transform_c.forward(cond)
    n = cond_z.shape[0]

    with torch.no_grad():
        if model_type == "flow": x_pred_z = model.model.sample(n, cond_z)
        elif model_type == "ddpm": x_pred_z = model.sample(n, cond_z)
        elif model_type == "sde": x_pred_z = model.sample(n, cond_z)

    x_pred = pre_transform_x.reverse(x_pred_z).cpu().numpy()
    x_true_np = x_true.cpu().numpy()

    plot_predictions(x_true_np, x_pred,
                     os.path.join(out_dir, f"{model_type}_predictions.png"),
                     names=["x0","y0","E","theta","phi"])


# ---------------------------
# Main
# ---------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", choices=["flow","ddpm","sde"], default="flow")
    p.add_argument("--train_data_path", default="/n/home04/hhanif/swgo_input_files/train.pt")
    p.add_argument("--val_data_path",   default="/n/home04/hhanif/swgo_input_files/val.pt")
    p.add_argument("--test_data_path",  default="/n/home04/hhanif/swgo_input_files/test.pt")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    dm = SWGODataModule(train_data_path=args.train_data_path,
                        val_data_path=args.val_data_path,
                        test_data_path=args.test_data_path,
                        batch_size=args.batch_size)
    dm.setup()
    train_loader, val_loader, pre_transform_x, pre_transform_c = make_loaders(dm, args.batch_size)

    cond, x = next(iter(train_loader))
    print(f"Sanity check batch shapes -> cond: {cond.shape}, x: {x.shape}")  # should be [B,4], [B,5]

    if args.model_type == "flow":
        model = FlowLightningModule(lr=args.lr)
        trainer, loss_tracker = build_trainer(args.epochs, "flow", patience=20)
    elif args.model_type == "ddpm":
        model = DDPM_Lightning(lr=args.lr)
        trainer, loss_tracker = build_trainer(args.epochs, "ddpm", patience=10)
    elif args.model_type == "sde":
        model = SDE_Lightning(lr=args.lr)
        trainer, loss_tracker = build_trainer(args.epochs, "sde", patience=10)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    visualize.plot_losses(train_losses=loss_tracker.train_losses,
                          val_losses=loss_tracker.val_losses)
    plt.savefig(f"results/{args.model_type}_loss_curves.png"); plt.close()

    run_predictions_and_plots(args.model_type, model, pre_transform_x, pre_transform_c, args.test_data_path)


if __name__ == "__main__":
    main()

import torch
import torch.optim as optim
import pytorch_lightning as pl
from pytorch_lightning.callbacks import TQDMProgressBar, EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt

from swgo_trainv2 import  LossTracker
from xaims import flows, ddpm, sde, visualize,transforms
from data_loader import SWGODataModule

# ---------------------------
# Utilities
# ---------------------------

def pool_cond(cond: torch.Tensor) -> torch.Tensor:
    """Pool [B, Nunits, 4] -> [B, 4]."""
    if cond.dim() == 3:
        return cond.mean(dim=1)
    return cond

class ZScore:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, eps: float = 1e-8):
        self.mean = mean
        self.std = torch.clamp(std, min=eps)
    def fwd(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std
    def inv(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.std + self.mean

def compute_zscore(dataset):
    conds, xs = [], []
    # No longer need to loop, can operate on the whole dataset tensor at once
    # for cond, x in dataset:
    #     conds.append(pool_cond(cond).unsqueeze(0).float())
    #     xs.append(x.unsqueeze(0).float())
    # cond_all = torch.cat(conds, dim=0)
    # x_all = torch.cat(xs, dim=0)

    # A much more efficient implementation
    all_data = [item for item in dataset]
    cond_all = torch.stack([pool_cond(item[0]).float() for item in all_data], dim=0)
    x_all = torch.stack([item[1].float() for item in all_data], dim=0)

    return ZScore(cond_all.mean(0, keepdim=True), cond_all.std(0, keepdim=True)), \
           ZScore(x_all.mean(0, keepdim=True),   x_all.std(0, keepdim=True))

class ZScoreWrappedDataset(torch.utils.data.Dataset):
    def __init__(self, base, zc: ZScore, zx: ZScore):
        self.base = base
        self.zc = zc
        self.zx = zx

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        cond, x = self.base[idx]

        # 🔧 remove extra singleton dims
        cond = cond.squeeze(0)   # e.g. [1, 90, 4] -> [90, 4]
        x = x.squeeze(0)         # e.g. [1, 5]    -> [5]

        cond = pool_cond(cond).float()  # [90,4] -> [4]
        x = x.float()                   # [5]

        return self.zc.fwd(cond), self.zx.fwd(x)




def make_loaders(dm: SWGODataModule, batch_size: int):
    zc, zx = compute_zscore(dm.train_dataset)
    train_ds = ZScoreWrappedDataset(dm.train_dataset, zc, zx)
    val_ds   = ZScoreWrappedDataset(dm.val_dataset,   zc, zx)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader   = torch.utils.data.DataLoader(val_ds,   batch_size=batch_size*2, shuffle=False, num_workers=1)
    return train_loader, val_loader, zc, zx


# ---------------------------
# Lightning wrappers
# ---------------------------

class FlowLightningModule(pl.LightningModule):
    def __init__(self, lr=1e-3, weight_decay=1e-5):
        super().__init__()
        self.save_hyperparameters()
        self.model = flows.CouplingFlow(
            x_dim=5, cond_dim=4,
            flow_layers=4,
            flow_type="spline",
            permute=True,
            nn_param=[
                {"hidden_dim": [96, 96], "layer_norm": True, "act": "silu", "dropout": 0.01},
                {"hidden_dim": [96, 96], "layer_norm": True, "act": "silu", "dropout": 0.01},
            ],
        )
    def training_step(self, batch, _):
        cond, x = batch
        loss = -self.model.log_prob(x, cond).mean()
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    def validation_step(self, batch, _):
        cond, x = batch
        loss = -self.model.log_prob(x, cond).mean()
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss
    def configure_optimizers(self):
        opt = optim.AdamW(self.model.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=20, min_lr=1e-5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}

class DDPM_Lightning(pl.LightningModule):
    def __init__(self, lr=1e-3, weight_decay=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = ddpm.DDPM(x_dim=5, cond_dim=4,
                               beta_start=1e-4, beta_end=0.02, diffusion_steps=1000,
                               time_embed_dim=8, nnet=None)
    def training_step(self, batch, _):
        cond, x = batch
        loss = self.model.loss(x, cond)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True); return loss
    def validation_step(self, batch, _):
        cond, x = batch
        loss = self.model.loss(x, cond)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True); return loss
    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=10, min_lr=1e-5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}
    def sample(self, n, cond): return self.model.sample(n, cond)

class SDE_Lightning(pl.LightningModule):
    def __init__(self, lr=1e-3, weight_decay=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = sde.SDEModel(
            sde=sde.VPSDE(beta_0=0.1, beta_1=20.0),
            x_dim=5, cond_dim=4, time_embed_dim=8,
            loss_weighting=False, nnet=None)
    def training_step(self, batch, _):
        cond, x = batch
        loss = self.model.loss(x, cond)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True); return loss
    def validation_step(self, batch, _):
        cond, x = batch
        loss = self.model.loss(x, cond)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True); return loss
    def configure_optimizers(self):
        opt = optim.AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        sch = optim.lr_scheduler.ReduceLROnPlateau(opt, "min", factor=0.5, patience=10, min_lr=1e-5)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "monitor": "val_loss"}}
    def sample(self, n, cond, steps=1000, use_ode=False): return self.model.sample(n, cond, steps, use_ode)


# ---------------------------
# Training / Evaluation
# ---------------------------

def build_trainer(epochs: int, ckpt_prefix: str, patience: int = 20):
    loss_tracker = LossTracker()
    callbacks = [
        TQDMProgressBar(refresh_rate=10),
        EarlyStopping(monitor="val_loss", mode="min", patience=patience),
        ModelCheckpoint(dirpath=f"checkpoints/{ckpt_prefix}",
                        filename="{epoch:02d}-{val_loss:.4f}",
                        monitor="val_loss", save_top_k=3, mode="min"),
        loss_tracker,
    ]
    trainer = pl.Trainer(max_epochs=epochs,
                         accelerator="gpu" if torch.cuda.is_available() else "cpu",
                         devices=1, callbacks=callbacks, log_every_n_steps=50)
    return trainer, loss_tracker


def plot_predictions(y_true, y_pred, out_path, names=None):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    D = y_true.shape[1]; cols = min(3, D); rows = int(np.ceil(D / cols))
    plt.figure(figsize=(5*cols, 3.5*rows))
    for d in range(D):
        plt.subplot(rows, cols, d+1)
        plt.hist(y_true[:, d], bins=60, histtype="step", label="True", density=True)
        plt.hist(y_pred[:, d], bins=60, histtype="step", label="Pred", density=True)
        plt.title(names[d] if names and d < len(names) else f"dim {d}")
        plt.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(out_path); plt.close()


def run_predictions_and_plots(model_type, model, zc, zx, test_path, out_dir="results"):
    test_inputs, test_labels = torch.load(test_path)
    cond = pool_cond(test_inputs.float()); x_true = test_labels.float()
    cond_z = zc.fwd(cond); n = cond_z.shape[0]
    with torch.no_grad():
        if model_type == "flow": x_pred_z = model.model.sample(n, cond_z)
        elif model_type == "ddpm": x_pred_z = model.sample(n, cond_z)
        elif model_type == "sde": x_pred_z = model.sample(n, cond_z)
    x_pred = zx.inv(x_pred_z).cpu().numpy(); x_true_np = x_true.cpu().numpy()
    plot_predictions(x_true_np, x_pred,
                     os.path.join(out_dir, f"{model_type}_predictions.png"),
                     names=["x0","y0","E","theta","phi"])


# ---------------------------
# Main
# ---------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", choices=["flow","ddpm","sde"], default="flow")
    p.add_argument("--train_data_path", default="/n/home04/hhanif/swgo_input_files/proper_train.pt")
    p.add_argument("--val_data_path",   default="/n/home04/hhanif/swgo_input_files/proper_val.pt")
    p.add_argument("--test_data_path",  default="/n/home04/hhanif/swgo_input_files/proper_test.pt")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    dm = SWGODataModule(train_data_path=args.train_data_path,
                        val_data_path=args.val_data_path,
                        test_data_path=args.test_data_path,
                        batch_size=args.batch_size)
    dm.setup()
    train_loader, val_loader, zc, zx = make_loaders(dm, args.batch_size)

    # 🔎 Sanity check
    cond, x = next(iter(train_loader))
    print(f"Sanity check batch shapes -> cond: {cond.shape}, x: {x.shape}")
    # Expect cond [B,4], x [B,5]

    if args.model_type == "flow":
        model = FlowLightningModule(lr=args.lr)
        trainer, loss_tracker = build_trainer(args.epochs, "flow", patience=20)
    elif args.model_type == "ddpm":
        model = DDPM_Lightning(lr=args.lr)
        trainer, loss_tracker = build_trainer(args.epochs, "ddpm", patience=10)
    elif args.model_type == "sde":
        model = SDE_Lightning(lr=args.lr)
        trainer, loss_tracker = build_trainer(args.epochs, "sde", patience=10)

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    visualize.plot_losses(train_losses=loss_tracker.train_losses,
                          val_losses=loss_tracker.val_losses)
    plt.savefig(f"results/{args.model_type}_loss_curves.png"); plt.close()

    run_predictions_and_plots(args.model_type, model, zc, zx, args.test_data_path)


if __name__ == "__main__":
    main()
