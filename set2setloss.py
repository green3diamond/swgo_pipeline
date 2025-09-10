import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment


class Set2SetLoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config
        self.regression_loss = nn.MSELoss(reduction="none")

    def forward(self, input, target, mask=None):
        """
        input:  (B, N, 5) predicted set (x0, y0, E, theta, phi)
        target: (B, N, 5) true set (same order)
        mask:   ignored (kept for API compatibility)
        """

        B, N, D = input.shape
        cost_mat = torch.cdist(input, target, p=2)  # (B, N, N)

        # Hungarian assignment per batch
        aligned_pred = []
        aligned_true = []
        for b in range(B):
            row_ind, col_ind = linear_sum_assignment(cost_mat[b].detach().cpu().numpy())
            aligned_pred.append(input[b, row_ind, :])
            aligned_true.append(target[b, col_ind, :])
        aligned_pred = torch.stack(aligned_pred, dim=0)  # (B, N, 5)
        aligned_true = torch.stack(aligned_true, dim=0)

        # --- Feature-wise losses ---
        x_loss     = (aligned_pred[..., 0] - aligned_true[..., 0]) ** 2
        y_loss     = (aligned_pred[..., 1] - aligned_true[..., 1]) ** 2
        E_loss     = (aligned_pred[..., 2] - aligned_true[..., 2]) ** 2
        theta_loss = (aligned_pred[..., 3] - aligned_true[..., 3]) ** 2

        # φ loss with wrapping (ensures periodicity)
        phi_diff   = aligned_pred[..., 4] - aligned_true[..., 4]
        phi_diff   = torch.remainder(phi_diff + torch.pi, 2 * torch.pi) - torch.pi
        phi_loss   = phi_diff ** 2

        # --- Mean over all elements ---
        x_loss     = x_loss.mean()
        y_loss     = y_loss.mean()
        E_loss     = E_loss.mean()
        theta_loss = theta_loss.mean()
        phi_loss   = phi_loss.mean()

        total_loss = x_loss + y_loss + E_loss + theta_loss + phi_loss

        return {
            "total_loss": total_loss.to(input.device),
            "x_loss": x_loss,
            "y_loss": y_loss,
            "E_loss": E_loss,
            "theta_loss": theta_loss,
            "phi_loss": phi_loss,
        }
