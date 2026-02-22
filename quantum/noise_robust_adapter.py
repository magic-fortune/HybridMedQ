import torch
import torch.nn as nn

class NoiseRobustAdapter(nn.Module):
    """
    Medical-noise robust adapter.
    Input:  features [B, D]  (来自 CNN 的局部/全局特征)
    Output: gated features [B, D], and gate g [B,1]
    """
    def __init__(self, feat_dim: int, hidden: int = 128):
        super().__init__()
        # gate network: 预测每个样本的“可靠性/清晰度”
        self.gate_net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),   # g in (0,1)
        )

        # optional: feature denoise projection（轻量去噪映射）
        self.denoise = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.LayerNorm(feat_dim),
        )

    def forward(self, x: torch.Tensor):
        # x: [B,D]
        x_dn = self.denoise(x)
        g = self.gate_net(x_dn)           # [B,1]
        x_gated = x_dn * g                # 门控抑制 noisy 特征幅度
        return x_gated, g
