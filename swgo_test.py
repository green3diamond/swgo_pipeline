import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
from pytorch_lightning import LightningModule
from swgo_trainv2 import CFMLightningModule

# --- Configuration ---
DATA_FILE_PATH = '/n/home04/hhanif/swgo_input_files/test.pt'
CKPT_PATH = '/n/home04/hhanif/swgo_training/swgo_trainv2_run_results_20250909_225355/epoch=031-val_loss=0.1037-train_loss_epoch=0.1708.ckpt'
OUTPUT_DIR = "analysis_plots"
THETA_MAX = torch.pi / 2  # Max theta value in radians

BATCH_SIZE = 512



# --- Denormalization ---
def denormalize_labels(norm_params: torch.Tensor, theta_max: float) -> torch.Tensor:
    x0_norm, y0_norm, e_norm, th_norm, ph_norm = norm_params.T
    x0 = x0_norm * 5000
    y0 = y0_norm * 5000
    e = 0.1 + (e_norm + 1) * (10 - 0.1) / 2
    theta = (th_norm + 1) * theta_max / 2
    phi = ph_norm * torch.pi
    return torch.stack([x0, y0, e, theta, phi], dim=1)


# --- Plotting ---
def plot_parameter_comparison(true_vals, pred_vals, label, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(f'Parameter Comparison for {label}', fontsize=18)

    # Histogram
    ax1.hist(true_vals, bins=60, label='True', density=True, histtype='step', linewidth=2)
    ax1.hist(pred_vals, bins=60, label='Predicted', density=True, histtype='step', linewidth=2)
    ax1.set_title('Distribution Comparison', fontsize=14)
    ax1.set_xlabel(label, fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.legend()
    ax1.grid(True, linestyle='--')

    # 2D Histogram
    plot_subset = min(10000, len(true_vals))
    indices = np.random.choice(len(true_vals), plot_subset, replace=False)
    hb = ax2.hist2d(true_vals[indices], pred_vals[indices], bins=50, cmap='viridis', cmin=1)
    ax2.set_title('Predicted vs. True Values', fontsize=14)
    ax2.set_xlabel(f'True {label}', fontsize=12)
    ax2.set_ylabel(f'Predicted {label}', fontsize=12)
    fig.colorbar(hb[3], ax=ax2, label='Count')

    lims = [np.min([ax2.get_xlim(), ax2.get_ylim()]),
            np.max([ax2.get_xlim(), ax2.get_ylim()])]
    ax2.plot(lims, lims, 'r--', alpha=0.8, label='Ideal (y=x)')
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.set_aspect('equal', adjustable='box')
    ax2.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(output_path, dpi=300)
    print(f"Saved {label.split(' ')[0]} comparison plot to: '{output_path}'")
    plt.close(fig)


# --- Main ---
def main():
    print("🚀 Starting analysis script...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load test data
    print(f"\nLoading test data from: '{DATA_FILE_PATH}'")
    test_inputs, test_labels = torch.load(DATA_FILE_PATH, map_location='cpu')
    test_dataset = TensorDataset(test_inputs, test_labels)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"Loaded {len(test_labels)} samples.")

    # 2. Load trained model from checkpoint
    print(f"\nLoading model from checkpoint: '{CKPT_PATH}'")
    model = CFMLightningModule.load_from_checkpoint(CKPT_PATH, input_dim=test_inputs.shape[1], output_dim=test_labels.shape[1])
    model.eval()

    # 3. Run predictions
    print("Running model inference on test set...")
    preds = []
    model.eval()

    with torch.no_grad():
        for batch in test_loader:
            cond, _ = batch  # your dataset = (cond, labels)
            # Use predict_step logic directly
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


    # 4. Denormalize
    print("Denormalizing parameters...")
    true_params_denorm = denormalize_labels(test_labels, THETA_MAX).numpy()
    pred_params_denorm = denormalize_labels(pred_params_norm, THETA_MAX).numpy()

    # 5. Generate plots
    print("\nGenerating parameter comparison plots...")
    param_info = [
        {'label': 'x0 (m)', 'name': 'x0'},
        {'label': 'y0 (m)', 'name': 'y0'},
        {'label': 'Energy (TeV)', 'name': 'energy'},
        {'label': 'Theta (rad)', 'name': 'theta'},
        {'label': 'Phi (rad)', 'name': 'phi'}
    ]

    for i, info in enumerate(param_info):
        plot_parameter_comparison(
            true_vals=true_params_denorm[:, i],
            pred_vals=pred_params_denorm[:, i],
            label=info['label'],
            output_path=os.path.join(OUTPUT_DIR, f"{info['name']}_comparison.png")
        )

    print(f"\n✅ All parameter comparison plots saved in '{OUTPUT_DIR}'")


if __name__ == "__main__":
    main()
