"""Collection of pytorch modules that make up the common networks used in my
projects."""

import math
from typing import Optional, Union

import torch as T
import torch.nn as nn
from torch_utils import get_act, get_nrm
from torch import Tensor, BoolTensor
import torch 


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        """Faster LayerNorm by seting elementwise_affine=False."""
        super().__init__(*args, **kwargs, elementwise_affine=False)


class RMSNorm(nn.Module):
    def __init__(self, dim: int):
        """RNMSNorm from https://arxiv.org/abs/1910.07467. Slower than LayerNorm."""
        super().__init__()
        self.scale = dim**0.5
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return F.normalize(x, dim=-1) * self.scale * self.weight
        return self._norm(x.float()).type_as(x) * self.weight

    def __repr__(self):
        return f"RMSNorm(dim={self.weight.shape[0]})"


def modulate(x, shift, scale):
    """Applies adaptive layer normalization modulation."""
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# --- Dense Network Modules ---
class MLPBlock(nn.Module):
    """A simple MLP block that makes up a dense network.

    Made up of several layers containing:
    - linear map
    - activation function [Optional]
    - layer normalisation [Optional]
    - dropout [Optional]

    Only the input of the block is concatentated with context information.
    For residual blocks, the input is added to the output of the final layer.
    """

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int,
        ctxt_dim: int = 0,
        n_layers: int = 1,
        act: str = "relu",
        nrm: str = "none",
        drp: float = 0,
        do_res: bool = False,
        do_bayesian: bool = False,
        init_zeros: bool = False,
    ) -> None:
        """Init method for MLPBlock."""
        super().__init__()

        self.inpt_dim = inpt_dim
        self.outp_dim = outp_dim
        self.ctxt_dim = ctxt_dim
        self.do_res = do_res and (inpt_dim == outp_dim)

        self.block = nn.ModuleList()
        for n in range(n_layers):
            lyr_in = inpt_dim + ctxt_dim if n == 0 else outp_dim
            self.block.append(nn.Linear(lyr_in, outp_dim))
            if init_zeros and n == n_layers - 1 and not do_bayesian:
                self.block[-1].weight.data.fill_(0)
                self.block[-1].bias.data.fill_(0)
            if act != "none":
                self.block.append(get_act(act))
            if nrm != "none":
                self.block.append(get_nrm(nrm, outp_dim))
            if drp > 0:
                self.block.append(nn.Dropout(drp))

    def forward(self, inpt: Tensor, ctxt: Optional[Tensor] = None) -> Tensor:
        if self.ctxt_dim and ctxt is None:
            raise ValueError("Was expecting contextual information but none has been provided!")
        temp = torch.cat([inpt, ctxt], dim=-1) if self.ctxt_dim else inpt
        for layer in self.block:
            temp = layer(temp)
        if self.do_res:
            temp = temp + inpt
        return temp

    def __repr__(self) -> str:
        string = str(self.inpt_dim)
        if self.ctxt_dim:
            string += f"({self.ctxt_dim})"
        string += "->"
        string += "->".join([str(b).split("(", 1)[0] for b in self.block])
        string += "->" + str(self.outp_dim)
        if self.do_res:
            string += "(add)"
        return string


class DenseNetwork(nn.Module):
    """A dense neural network made from a series of consecutive MLP blocks and
    context injection layers."""

    def __init__(
        self,
        inpt_dim: int,
        outp_dim: int = 0,
        hddn_dim: Union[int, list] = 32,
        act_h: str = "relu",
        ctxt_dim: int = 0,
        num_blocks: int = 1,
        n_lyr_pbk: int = 1,
        act_o: str = "none",
        do_out: bool = True,
        nrm: str = "none",
        drp: float = 0,
        do_res: bool = False,
        ctxt_in_inpt: bool = True,
        ctxt_in_hddn: bool = False,
        do_bayesian: bool = False,
        output_init_zeros: bool = False,
    ) -> None:
        """Initialise the DenseNetwork."""
        super().__init__()

        if ctxt_dim and not ctxt_in_hddn and not ctxt_in_inpt:
            raise ValueError("Network has context inputs but nowhere to use them!")

        self.inpt_dim = inpt_dim
        if not isinstance(hddn_dim, int) and isinstance(hddn_dim, list):
            self.hddn_dim = hddn_dim
        else:
            self.hddn_dim = num_blocks * [hddn_dim]
        self.outp_dim = outp_dim or inpt_dim if do_out else self.hddn_dim[-1]
        self.num_blocks = len(self.hddn_dim)
        self.ctxt_dim = ctxt_dim
        self.do_out = do_out
        self.hidden_features = self.hddn_dim[-1]

        self.input_block = MLPBlock(
            inpt_dim=self.inpt_dim,
            outp_dim=self.hddn_dim[0],
            ctxt_dim=self.ctxt_dim if ctxt_in_inpt else 0,
            act=act_h, nrm=nrm, drp=drp
        )
        self.hidden_blocks = nn.ModuleList()
        if self.num_blocks > 1:
            for h_1, h_2 in zip(self.hddn_dim[:-1], self.hddn_dim[1:]):
                self.hidden_blocks.append(
                    MLPBlock(
                        inpt_dim=h_1, outp_dim=h_2,
                        ctxt_dim=self.ctxt_dim if ctxt_in_hddn else 0,
                        n_layers=n_lyr_pbk, act=act_h, nrm=nrm,
                        drp=drp, do_res=do_res
                    )
                )
        if do_out:
            self.output_block = MLPBlock(
                inpt_dim=self.hddn_dim[-1], outp_dim=self.outp_dim, act=act_o,
                do_bayesian=do_bayesian, init_zeros=output_init_zeros
            )

    def forward(self, inputs: Tensor, ctxt: Optional[Tensor] = None) -> Tensor:
        if ctxt is not None:
            dim_diff = inputs.dim() - ctxt.dim()
            if dim_diff > 0:
                ctxt = ctxt.view(ctxt.shape[0], *dim_diff * (1,), *ctxt.shape[1:])
                ctxt = ctxt.expand(*inputs.shape[:-1], -1)
        
        # In this specific script, context is handled outside the DenseNetwork.
        # This forward signature is for general use, but here `ctxt` will be None.
        temp = self.input_block(inputs, ctxt)
        for h_block in self.hidden_blocks:
            temp = h_block(temp, ctxt)
        if self.do_out:
            temp = self.output_block(temp)
        return temp


# --- Attention Module and Helpers from swgo_attention.py ---



# --- Core Modules from Files ---

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256, max_period=10000, time_factor: float = 1.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
        self.max_period = max_period
        self.time_factor = time_factor
        half = self.frequency_embedding_size // 2
        freqs = T.exp(-math.log(max_period) * T.arange(start=0, end=half, dtype=T.float32) / half)
        self.register_buffer("freqs", freqs, persistent=False)

    def timestep_embedding(self, t):
        t_scaled = t * self.time_factor
        args = t_scaled[:, None].float() * self.freqs[None]
        embedding = T.cat([T.cos(args), T.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = T.cat([embedding, T.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t).type_as(t)
        return self.mlp(t_freq)



class GLU(nn.Module):
    """Gated Linear Unit activation function."""
    def __init__(self, embed_dim: int, hidden_dim: int | None = None, activation: str = "SiLU",
                 dropout: float = 0.0, bias: bool = True, gated: bool = True):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = embed_dim * 2
        self.gated = gated
        self.in_proj = nn.Linear(embed_dim, hidden_dim * 2 if gated else hidden_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_dim, embed_dim, bias=bias)
        self.drop = nn.Dropout(dropout)
        self.activation = getattr(nn, activation)()

    def forward(self, x: Tensor) -> Tensor:
        x = self.in_proj(x)
        if self.gated:
            x1, x2 = x.chunk(2, dim=-1)
            x = self.activation(x1) * x2
        else:
            x = self.activation(x)
        x = self.drop(x)
        return self.out_proj(x)

class CA_DiT_Block(nn.Module):
    """Cross-Attention Diffusion Transformer Block."""
    def __init__(self, hidden_dim: int, num_heads: int, mha_config: dict = None, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_dim)
        self.norm_k = RMSNorm(hidden_dim)
        self.norm2 = RMSNorm(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = GLU(hidden_dim, mlp_hidden_dim, bias=False, gated=True)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, hidden_dim * 6, bias=True))
        if mha_config is None: mha_config = {}
        mha_config['do_selfattn'] = False
        self.attn = Attention(embed_dim=hidden_dim, num_heads=num_heads, **mha_config)

    def forward(self, x, y, c, mask_y=None):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        # For kv_mask, PyTorch SDPA expects True for values to be masked.
        x = x + gate_msa.unsqueeze(1) * self.attn(x=modulate(self.norm1(x), shift_msa, scale_msa), kv=self.norm_k(y), kv_mask=mask_y)
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

    def reset_parameters(self):
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)


class ResNetBlock(nn.Module):
    """Simple residual block with 2-layer MLP expansion + dropout."""
    def __init__(self, model_dim, dropout=0.1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, model_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim * 4, model_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + self.mlp(x)


class CFMDiagramModel(nn.Module):
    """
    CFM model aligned to diagram:
      - sine timestep embedding + MLP
      - separate embeddings for conditioned & noisy target features
      - concatenated time+cond context 'C'
      - adaLN-zero
      - stack of residual blocks
      - final MLP outputs vector field

    data_dim: number of target features (x0, y0, E, theta, phi) -> 5
    cond_feature_dim: number of conditioned features (x, y, N, T) -> 4
    """
    def __init__(self, data_dim=5, cond_feature_dim=4, model_dim=128, num_res_blocks=3, dropout=0.1):
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

        # Condition features: support (B, 4) or (B, M, 4) for detector-level cond=(x,y,N,T)
        cond_emb = self.cond_mlp(cond)
        if cond_emb.dim() == 3:
            # (B, M, H) -> pool across detectors
            pooled_cond = cond_emb.mean(dim=1)
        elif cond_emb.dim() == 2:
            # (B, H) -> already pooled / single detector vector
            pooled_cond = cond_emb
        else:
            raise ValueError(f"Unexpected cond_emb shape {cond_emb.shape}")

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

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return self.weight * x

class SwiGLU(nn.Module):
    """SwiGLU = (XW1) * SiLU(XW2). Output dim equals hidden_dim."""
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim)
        self.w2 = nn.Linear(in_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w1(x) * torch.nn.functional.silu(self.w2(x))



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

class SelfAttentionBlock(nn.Module):
    """
    Lightweight Transformer-style block:
      LN -> MHA (batch_first) -> residual
      LN -> SwiGLU FFN -> residual
    """
    def __init__(self, model_dim: int, num_heads: int = 8, dropout: float = 0.1, use_rmsnorm: bool = True, ffn_mult: int = 4):
        super().__init__()
        Norm = RMSNorm if use_rmsnorm else nn.LayerNorm
        self.norm1 = Norm(model_dim)
        self.attn = nn.MultiheadAttention(model_dim, num_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = Norm(model_dim)
        self.ffn = nn.Sequential(
            SwiGLU(model_dim, model_dim * ffn_mult),
            nn.Dropout(dropout),
            nn.Linear(model_dim * ffn_mult, model_dim),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H) -> treat as sequence length 1
        y = self.norm1(x)
        y = y.unsqueeze(1)                          # (B, 1, H)
        attn, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop1(attn.squeeze(1))         # residual

        y = self.norm2(x)
        y = self.ffn(y)
        x = x + self.drop2(y)
        return x

# ==============================
#   Normalization Utils
# ==============================
class StdNormalizer:
    def __init__(self, name=""):
        self.name = name
        self.mean = None
        self.std = None

    def calculate(self, x: torch.Tensor):
        mean = x.mean()
        if x.numel() < 2:
            std = torch.tensor(0.1).float() if self.name in ["eta", "phi"] else torch.tensor(1.0).float()
        else:
            std = x.std()
            if std == 0:
                std = torch.tensor(0.1).float() if self.name in ["eta", "phi"] else torch.tensor(1.0).float()
        self.mean = mean
        self.std = std
        return mean, std

    def forward(self, x: torch.Tensor):
        if self.mean is None or self.std is None:
            self.calculate(x)
        return (x - self.mean) / self.std

    def inverse(self, x: torch.Tensor):
        if self.mean is None or self.std is None:
            raise RuntimeError("StdNormalizer not fitted yet.")
        return x * self.std + self.mean
