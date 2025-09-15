import pytorch_lightning as pl
import numpy as np
import torch
import os
import h5py
from torch.utils.data import DataLoader, TensorDataset
from swgo_trainv2 import CFMLightningModule
import matplotlib.pyplot as plt

# --- Configuration ---
CKPT_PATH = "/n/home04/hhanif/test15sept/swgo_trainv2_run_results_20250915_045024/ckpts/epoch=319.ckpt"
BATCH_SIZE = 512





class SWGODataModule(pl.LightningDataModule):
    """
    Loads test dataset from .h5, applies normalization, and builds a test dataloader.
    """
    def __init__(self, 
                 test_h5_path="/n/home04/hhanif/swgo_input_files/test_perfect.h5",
                 batch_size=256,
                 seed=42):
        super().__init__()
        self.test_h5_path = test_h5_path
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = 10

    def _zscore(self, data: np.ndarray, name: str):
        mean = data.mean()
        std = data.std()
        if std == 0:
            std = 0.1 if name in ["phi"] else 1.0
        return (data - mean) / std

    def _normalize(self, inputs, labels):
        # Normalize inputs (unchanged)
        inputs_norm = np.zeros_like(inputs)
        names_inputs = ["x", "y", "N", "T"]
        for i, name in enumerate(names_inputs):
            arr = inputs[:, :, i]
            if name in ["E", "N"]:  # log-transform for E & N
                arr = np.log(arr + 1e-8)
            inputs_norm[:, :, i] = self._zscore(arr, name)

        # Normalize labels and save stats
        labels_norm = np.zeros_like(labels)
        names_labels = ["x0", "y0", "E", "theta", "phi"]

        self.label_stats = {}   # <--- add this

        for i, name in enumerate(names_labels):
            arr = labels[:, i]
            if name in ["E", "N"]:
                arr = np.log(arr + 1e-8)
            mean, std = arr.mean(), arr.std()
            if std == 0:
                std = 0.1 if name == "phi" else 1.0
            labels_norm[:, i] = (arr - mean) / std
            self.label_stats[name] = (mean, std)   # <--- save for later

        return inputs_norm, labels_norm



    # --- Denormalization (from your training) ---
    def denormalize(self, labels_norm):
        labels_denorm = np.zeros_like(labels_norm)
        names_labels = ["x0", "y0", "E", "theta", "phi"]
        for i, name in enumerate(names_labels):
            mean, std = self.label_stats[name]
            arr = labels_norm[:, i] * std + mean
            if name in ["E", "N"]:
                arr = np.exp(arr)  # undo log
            labels_denorm[:, i] = arr
        return labels_denorm
    def setup(self, stage=None):
        # --- Test data only ---
        with h5py.File(self.test_h5_path, "r") as f:
            inputs = f["inputs"][:]   # (events, nunits, 4)
            labels = f["labels"][:]   # (events, 5)

        inputs_norm, labels_norm = self._normalize(inputs, labels)

        inputs_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        labels_tensor = torch.tensor(labels_norm, dtype=torch.float32)

        self.test_dataset = TensorDataset(inputs_tensor, labels_tensor)

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size * 2,
            num_workers=self.num_workers,
            pin_memory=True
        )


# --- Plotting ---
def plot_distribution(true_vals, pred_vals, label, output_path, xlim=None):
    """Plot histogram distributions of true vs predicted."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.hist(true_vals, bins=60, label="True", density=True, histtype="step", linewidth=2)
    ax.hist(pred_vals, bins=60, label="Predicted", density=True, histtype="step", linewidth=2)
    ax.set_title(f"Distribution Comparison for {label}", fontsize=14)
    ax.set_xlabel(label, fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle="--")
    if xlim:
        ax.set_xlim(xlim)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved distribution plot to: '{output_path}'")
    plt.close(fig)


def plot_comparison(true_vals, pred_vals, label, output_path, xlim=None):
    """Plot 2D scatter/hist2d comparison between true and predicted."""
    fig, ax = plt.subplots(figsize=(8, 7))

    plot_subset = min(10000, len(true_vals))
    indices = np.random.choice(len(true_vals), plot_subset, replace=False)
    hb = ax.hist2d(true_vals[indices], pred_vals[indices], bins=50, cmap="viridis", cmin=1)
    ax.set_title(f"Predicted vs. True {label}", fontsize=14)
    ax.set_xlabel(f"True {label}", fontsize=12)
    ax.set_ylabel(f"Predicted {label}", fontsize=12)
    fig.colorbar(hb[3], ax=ax, label="Count")

    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()])]
    ax.plot(lims, lims, "r--", alpha=0.8, label="Ideal (y=x)")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    if xlim:
        ax.set_xlim(xlim)
        ax.set_ylim(xlim)
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    print(f"Saved comparison plot to: '{output_path}'")
    plt.close(fig)





# --- Main ---
def main():
    print("🚀 Starting test evaluation...")

    # Save folders
    CKPT_DIR = os.path.dirname(CKPT_PATH)
    DIST_DIR = os.path.join(CKPT_DIR, "distribution")
    COMP_DIR = os.path.join(CKPT_DIR, "comparison")
    os.makedirs(DIST_DIR, exist_ok=True)
    os.makedirs(COMP_DIR, exist_ok=True)

    # 1. Load test dataset via DataModule
    dm = SWGODataModule(batch_size=BATCH_SIZE)
    dm.setup()
    test_loader = dm.test_dataloader()
    print(f"Loaded {len(dm.test_dataset)} test samples.")

    # 2. Load trained model from checkpoint
    print(f"\nLoading model from checkpoint: '{CKPT_PATH}'")
    sample_inputs, sample_labels = next(iter(test_loader))
    model = CFMLightningModule.load_from_checkpoint(
        CKPT_PATH,
        input_dim=sample_inputs.shape[1],
        output_dim=sample_labels.shape[1]
    )
    model.eval()

    # 3. Run predictions
    print("Running model inference on test set...")
    preds = []
    with torch.no_grad():
        for batch in test_loader:
            cond, _ = batch
            num_steps = 100
            x = torch.randn(cond.size(0), 5, device=model.device)
            time_steps = torch.linspace(0, 1, num_steps + 1, device=model.device)
            step_size = time_steps[1] - time_steps[0]
            for i in range(num_steps):
                t = time_steps[i]
                v_pred = model(x, cond.to(model.device), t.repeat(cond.size(0)))
                x = x + v_pred * step_size
            preds.append(x.cpu())

    pred_params_norm = torch.cat(preds, dim=0)


    # Convert tensors to numpy
    true_params_norm = torch.cat([batch[1] for batch in test_loader], dim=0).numpy()
    pred_params_norm = pred_params_norm.numpy()

    # Denormalize using stored stats
    true_params_denorm = dm.denormalize(true_params_norm)
    pred_params_denorm = dm.denormalize(pred_params_norm)


    # 5. Generate plots
    print("\nGenerating parameter comparison and distribution plots...")
    param_info = [
        {"label": "x0 (m)", "name": "x0"},
        {"label": "y0 (m)", "name": "y0"},
        {"label": "Energy (TeV)", "name": "energy"},
        {"label": "Theta (rad)", "name": "theta"},
        {"label": "Phi (rad)", "name": "phi"}
    ]

    for i, info in enumerate(param_info):
        if info["name"] == "energy":
            xlim = (0, 2.5)
        else:
            xlim = None

        # Distribution plot
        dist_path = os.path.join(DIST_DIR, f"{info['name']}_distribution.png")
        plot_distribution(
            true_vals=true_params_denorm[:, i],
            pred_vals=pred_params_denorm[:, i],
            label=info["label"],
            output_path=dist_path,
            xlim=xlim
        )

        # Comparison plot
        comp_path = os.path.join(COMP_DIR, f"{info['name']}_comparison.png")
        plot_comparison(
            true_vals=true_params_denorm[:, i],
            pred_vals=pred_params_denorm[:, i],
            label=info["label"],
            output_path=comp_path,
            xlim=xlim
        )

    print(f"\n✅ All plots saved inside:\n  - {DIST_DIR}\n  - {COMP_DIR}")


if __name__ == "__main__":
    main()
