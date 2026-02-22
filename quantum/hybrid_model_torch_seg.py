import math

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    import torchquantum as tq
    import torchquantum.functional as tqf
    from torchquantum.measurement import expval_joint_analytical
except Exception as exc:
    raise ImportError(
        "TorchQuantum is required for hybrid_model_torch_v5_seg.py. "
        "Please install torchquantum before using this model."
    ) from exc


class QuantumLayerTorchQuantumSeg(nn.Module):
    def __init__(
        self,
        n_qubits: int = 8,
        q_layers: int = 2,
        entangle: str = "ring",
        measure_z: bool = True,
        measure_zz: bool = True,
        measure_xx: bool = False,
        correlator_pairs: str = "ring",
        reupload: bool = True,
        input_scaling_init: float = math.pi,
        use_readout: bool = True,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_layers = q_layers
        self.entangle = entangle
        self.measure_z = measure_z
        self.measure_zz = measure_zz
        self.measure_xx = measure_xx
        self.correlator_pairs = correlator_pairs
        self.reupload = reupload
        self.use_readout = use_readout

        self.weights = nn.Parameter(0.01 * torch.randn(q_layers, n_qubits, 3, dtype=torch.float32))
        self.input_scale = nn.Parameter(
            torch.full((n_qubits,), float(input_scaling_init), dtype=torch.float32)
        )
        self.reupload_scale = nn.Parameter(0.1 * torch.randn(q_layers, n_qubits, dtype=torch.float32))
        self.reupload_shift = nn.Parameter(torch.zeros(q_layers, n_qubits, dtype=torch.float32))
        self.readout_weights = nn.Parameter(0.01 * torch.randn(n_qubits, 3, dtype=torch.float32))

        self.measure_all = tq.MeasureAll(tq.PauliZ)
        self.pairs = self._build_pairs()

    def _entangle_block(self, qdev: "tq.QuantumDevice"):
        q = self.n_qubits
        if q < 2:
            return
        if self.entangle == "ring":
            for i in range(q):
                tqf.cnot(qdev, wires=[i, (i + 1) % q])
        elif self.entangle == "local":
            for i in range(q - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
        elif self.entangle == "full":
            for i in range(q):
                for j in range(i + 1, q):
                    tqf.cnot(qdev, wires=[i, j])
        else:
            raise ValueError(f"Unknown entangle='{self.entangle}'")

    def _build_pairs(self):
        q = self.n_qubits
        if q < 2:
            return []
        if self.correlator_pairs == "ring":
            return [(i, (i + 1) % q) for i in range(q)]
        if self.correlator_pairs == "full":
            pairs = []
            for i in range(q):
                for j in range(i + 1, q):
                    pairs.append((i, j))
            return pairs
        raise ValueError(f"Unknown correlator_pairs='{self.correlator_pairs}'")

    def _pauli_string(self, terms):
        s = ["I"] * self.n_qubits
        for wire, pauli in terms:
            s[wire] = pauli
        return "".join(s)

    @property
    def out_dim(self):
        d = 0
        if self.measure_z:
            d += self.n_qubits
        if self.measure_zz:
            d += len(self.pairs)
        if self.measure_xx:
            d += len(self.pairs)
        return d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(dtype=torch.float32)
        bsz = x.shape[0]

        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=x.device)

        for i in range(self.n_qubits):
            tqf.ry(qdev, wires=i, params=x[:, i] * self.input_scale[i])

        for l in range(self.q_layers):
            if self.reupload:
                for i in range(self.n_qubits):
                    tqf.rx(
                        qdev,
                        wires=i,
                        params=x[:, i] * self.reupload_scale[l, i] + self.reupload_shift[l, i],
                    )
            for i in range(self.n_qubits):
                params = self.weights[l, i].unsqueeze(0).repeat(bsz, 1)
                tqf.rot(qdev, wires=i, params=params)
            self._entangle_block(qdev)

        if self.use_readout:
            for i in range(self.n_qubits):
                params = self.readout_weights[i].unsqueeze(0).repeat(bsz, 1)
                tqf.rot(qdev, wires=i, params=params)

        obs = []
        if self.measure_z:
            z_all = self.measure_all(qdev)
            for i in range(self.n_qubits):
                obs.append(z_all[:, i])

        if self.measure_zz:
            for (i, j) in self.pairs:
                obs.append(expval_joint_analytical(qdev, self._pauli_string([(i, "Z"), (j, "Z")])))

        if self.measure_xx:
            for (i, j) in self.pairs:
                obs.append(expval_joint_analytical(qdev, self._pauli_string([(i, "X"), (j, "X")])))

        return torch.stack(obs, dim=1).to(dtype=torch.float32)


class ImageToQubits(nn.Module):
    def __init__(self, in_channels: int, patch_size: int, n_qubits: int, hidden: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(hidden * 4 * 4, n_qubits),
            nn.LayerNorm(n_qubits),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        _, _, h, w = image.shape
        ps = self.patch_size
        if h == ps and w == ps:
            patch = image
        else:
            patch = F.interpolate(image, size=(ps, ps), mode="bilinear", align_corners=False)
        return self.enc(patch)


class SpatialConcatFusion(nn.Module):
    def __init__(
        self,
        cnn_channels: int,
        q_dim: int,
        q_hidden: int = 256,
        use_gate: bool = True,
        gate_hidden: int = 128,
        init_gate_bias: float = -2.0,
    ):
        super().__init__()
        self.q_to_cnn = nn.Sequential(
            nn.Linear(q_dim, q_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(q_hidden, cnn_channels),
        )
        self.use_gate = use_gate
        if self.use_gate:
            self.gate = nn.Sequential(
                nn.Linear(cnn_channels, gate_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(gate_hidden, 1),
            )
            nn.init.constant_(self.gate[-1].bias, init_gate_bias)

        self.mix = nn.Sequential(
            nn.Conv2d(cnn_channels * 2, cnn_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(cnn_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, cnn_feat: torch.Tensor, q_feat: torch.Tensor) -> torch.Tensor:
        bsz, channels, h, w = cnn_feat.shape
        q_proj = self.q_to_cnn(q_feat)
        if self.use_gate:
            cnn_vec = F.adaptive_avg_pool2d(cnn_feat, 1).view(bsz, channels)
            gate = torch.sigmoid(self.gate(cnn_vec))
            q_proj = gate * q_proj

        q_map = q_proj.view(bsz, channels, 1, 1).expand(-1, -1, h, w)
        fused = torch.cat([cnn_feat, q_map], dim=1)
        return self.mix(fused)


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels, dropout=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class HybridModel_TQ_Seg(nn.Module):
    def __init__(
        self,
        image_channels: int = 3,
        base_channels: int = 32,
        num_classes: int = 1,
        use_aspp: bool = False,
        aspp_rates=(2, 4, 6),
        aspp_dropout: float = 0.1,
        use_quantum: bool = True,
        n_qubits: int = 8,
        q_layers: int = 2,
        q_entangle: str = "ring",
        measure_z: bool = True,
        measure_zz: bool = True,
        measure_xx: bool = False,
        correlator_pairs: str = "ring",
        q_reupload: bool = True,
        q_input_scale: float = math.pi,
        q_use_readout: bool = True,
        q_patch_size: int = 32,
        q_enc_hidden: int = 128,
        q_hidden: int = 256,
        fusion_gate: bool = True,
        gate_hidden: int = 128,
        init_gate_bias: float = -1.0,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.use_quantum = use_quantum

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.enc1 = ConvBlock(image_channels, c1, dropout=0.0)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(c1, c2, dropout=0.0)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = ConvBlock(c2, c3, dropout=dropout)

        self.aspp = ASPP(
            in_channels=c3,
            out_channels=c3,
            atrous_rates=aspp_rates,
            dropout=aspp_dropout,
        ) if use_aspp else nn.Identity()

        if self.use_quantum:
            self.img2q = ImageToQubits(
                in_channels=image_channels,
                patch_size=q_patch_size,
                n_qubits=n_qubits,
                hidden=q_enc_hidden,
            )
            self.quantum = QuantumLayerTorchQuantumV5Seg(
                n_qubits=n_qubits,
                q_layers=q_layers,
                entangle=q_entangle,
                measure_z=measure_z,
                measure_zz=measure_zz,
                measure_xx=measure_xx,
                correlator_pairs=correlator_pairs,
                reupload=q_reupload,
                input_scaling_init=q_input_scale,
                use_readout=q_use_readout,
            )
            self.fusion = SpatialConcatFusion(
                cnn_channels=c3,
                q_dim=self.quantum.out_dim,
                q_hidden=q_hidden,
                use_gate=fusion_gate,
                gate_hidden=gate_hidden,
                init_gate_bias=init_gate_bias,
            )
        else:
            self.img2q = None
            self.quantum = None
            self.fusion = None

        self.up2 = UpBlock(c3, c2, c2, dropout=dropout)
        self.up1 = UpBlock(c2, c1, c1, dropout=dropout)
        self.seg_head = nn.Conv2d(c1, num_classes, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(image)
        x2 = self.enc2(self.pool1(x1))
        x3 = self.bottleneck(self.pool2(x2))
        x3 = self.aspp(x3)

        if self.use_quantum:
            q_in = self.img2q(image)
            q_feat = self.quantum(q_in)
            x3 = self.fusion(x3, q_feat)

        d2 = self.up2(x3, x2)
        d1 = self.up1(d2, x1)
        logits = self.seg_head(d1)

        if logits.shape[-2:] != image.shape[-2:]:
            logits = F.interpolate(logits, size=image.shape[-2:], mode="bilinear", align_corners=False)
        return logits
