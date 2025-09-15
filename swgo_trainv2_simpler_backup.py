import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from modules import TimestepEmbedder, ResNetBlock
from conditional_flow_matching import TargetConditionalFlowMatcher
from dpm import DPM_Solver, NoiseScheduleFlow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
#   Normalization
# ==============================
def normalize_std(x, name=""):
    mean = x.mean()
    if x.numel() < 2:
        std = torch.tensor(0.1).float() if name in ["eta", "phi"] else torch.tensor(1.0).float()
    else:
        std = x.std()
        if std == 0:
            std = torch.tensor(0.1).float() if name in ["eta", "phi"] else torch.tensor(1.0).float()
    return (x - mean) / std, mean, std

# ==============================
#   Load Data
# ==============================
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

x_dim = x_tensor_norm.shape[-1]
c_dim = c_tensor_norm.shape[-1]

# ==============================
#   Architecture
# ==============================
class CFMDiagramModel(nn.Module):
    def __init__(self, data_dim=x_dim, cond_feature_dim=c_dim, model_dim=128, num_res_blocks=3):
        super().__init__()
        self.time_embedder = TimestepEmbedder(model_dim,time_factor=1)

        # choose activation dynamically
        if act.lower() == "gelu":
            self.act = nn.GELU()
        elif act.lower() == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {act}")


    def two_layer_mlp(in_dim, out_dim, hidden_dims=None, activation=nn.GELU()):
        if hidden_dims is None:
            hidden_dims = [out_dim, out_dim]
        assert len(hidden_dims) == 2, "hidden_dims must have exactly two values"

        return nn.Sequential(
            nn.Linear(in_dim, hidden_dims[0]),
            activation,
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            activation,
            nn.Linear(hidden_dims[1], out_dim),
        )

        self.cond_mlp = two_layer_mlp(360, model_dim, hidden_dims=[256, 128], activation=self.act)
        self.noisy_target_mlp = two_layer_mlp(data_dim, model_dim, hidden_dims=[256, 128], activation=self.act)
        self.conditioning_C = two_layer_mlp(model_dim * 2, model_dim, hidden_dims=[256, 128], activation=self.act)

        # adaLN-zero
        self.adaLN_generator = nn.Linear(model_dim, model_dim * 2)
        self.layer_norm = nn.LayerNorm(model_dim, elementwise_affine=False)
        nn.init.zeros_(self.adaLN_generator.weight)
        nn.init.zeros_(self.adaLN_generator.bias)

        self.resnet_blocks = nn.ModuleList(
            [ResNetBlock(model_dim, dropout=0.0) for _ in range(num_res_blocks)]
        )

        self.final_mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            self.act,
            nn.Linear(model_dim * 4, model_dim),
        )
        self.output_layer = nn.Linear(model_dim, data_dim)

    def forward(self, x_t, cond, t):
        time_emb = self.time_embedder(t)

        cond_emb = self.cond_mlp(cond)


        combined = torch.cat([time_emb, pooled_cond], dim=-1)
        c = self.conditioning_C(combined)

        h = self.noisy_target_mlp(x_t)
        adaLN_params = self.adaLN_generator(c)
        scale, shift = adaLN_params.chunk(2, dim=-1)
        h = self.layer_norm(h) * (1 + scale) + shift

        for block in self.resnet_blocks:
            h = block(h)

        h = self.final_mlp(h)
        return self.output_layer(h)

# ==============================
#   Training Loop
# ==============================
model = CFMDiagramModel(num_res_blocks=4,act="gelu").to(device)
optimizer = optim.AdamW(model.parameters(), lr=3e-5)
scheduler_ddpm = optim.lr_scheduler.ReduceLROnPlateau(optimizer_ddpm, 'min', factor=0.5, patience=10, min_lr=1e-5)

criterion = nn.MSELoss()
flow_matcher = TargetConditionalFlowMatcher(sigma=1e-6)

for epoch in range(10):  # epochs
    model.train()
    total_loss = 0
    for cond, x1 in train_loader:
        cond, x1 = cond.to(device), x1.to(device)
        x1 = x1.view(x1.size(0), -1)
        x0 = torch.randn_like(x1)

        t, xt, ut = flow_matcher.sample_location_and_conditional_flow(x0, x1)
        v_pred = model(xt, cond, t)

        loss = criterion(v_pred, ut)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}: Train Loss = {avg_loss:.4f}")
    # step the scheduler with the metric you want to monitor
    scheduler_ddpm.step(avg_loss)