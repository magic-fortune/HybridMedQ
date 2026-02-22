import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from quantum.aspp import ASPP

try:
    import torchquantum as tq
    import torchquantum.functional as tqf
    from torchquantum.measurement import expval_joint_analytical
except Exception as exc:
    raise ImportError(
        "TorchQuantum is required for hybrid_model_torch_v5_tq.py. "
        "Please install torchquantum before using this model."
    ) from exc


# =========================================================
# 1) Quantum Layer (TorchQuantum implementation)
# =========================================================
class QuantumLayerTorchQuantum(nn.Module):
    def __init__(
        self,
        n_qubits: int = 8,
        q_layers: int = 2,
        entangle: str = "ring",          # "ring" | "full" | "local"
        measure_z: bool = True,
        measure_zz: bool = True,
        measure_xx: bool = False,
        correlator_pairs: str = "ring",  # "ring" | "full"
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

        # [L, Q, 3]
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
        Q = self.n_qubits
        if Q < 2:
            return
        if self.entangle == "ring":
            for i in range(Q):
                tqf.cnot(qdev, wires=[i, (i + 1) % Q])
        elif self.entangle == "local":
            for i in range(Q - 1):
                tqf.cnot(qdev, wires=[i, i + 1])
        elif self.entangle == "full":
            for i in range(Q):
                for j in range(i + 1, Q):
                    tqf.cnot(qdev, wires=[i, j])
        else:
            raise ValueError(f"Unknown entangle='{self.entangle}'")

    def _build_pairs(self):
        Q = self.n_qubits
        if Q < 2:
            return []
        if self.correlator_pairs == "ring":
            return [(i, (i + 1) % Q) for i in range(Q)]
        if self.correlator_pairs == "full":
            pairs = []
            for i in range(Q):
                for j in range(i + 1, Q):
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
        # x: [B, n_qubits]
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(dtype=torch.float32)
        bsz = x.shape[0]

        qdev = tq.QuantumDevice(n_wires=self.n_qubits, bsz=bsz, device=x.device)

        # encode
        for i in range(self.n_qubits):
            tqf.ry(qdev, wires=i, params=x[:, i] * self.input_scale[i])

        # layers
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
            z_all = self.measure_all(qdev)  # [B, n_qubits]
            for i in range(self.n_qubits):
                obs.append(z_all[:, i])

        if self.measure_zz:
            for (i, j) in self.pairs:
                obs.append(expval_joint_analytical(qdev, self._pauli_string([(i, "Z"), (j, "Z")])))

        if self.measure_xx:
            for (i, j) in self.pairs:
                obs.append(expval_joint_analytical(qdev, self._pauli_string([(i, "X"), (j, "X")])))

        return torch.stack(obs, dim=1).to(dtype=torch.float32)


# =========================================================
# 2) Image -> Quantum input (patch extractor + encoder)
# =========================================================
class ImageToQubits(nn.Module):
    """
    Make "image directly goes into quantum circuit" happen.

    Strategy:
      - take a fixed-size center patch from image: [B,C,patch,patch]
      - lightweight conv encoder -> vector -> n_qubits (sigmoid)
    """
    def __init__(self, in_channels: int, patch_size: int, n_qubits: int, hidden: int = 64):
        super().__init__()
        self.patch_size = patch_size
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),   # force small
            nn.Flatten(),
            nn.Linear(hidden * 4 * 4, n_qubits),
            nn.LayerNorm(n_qubits),
            nn.Tanh(),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        # image: [B,C,H,W]
        _, _, H, W = image.shape
        ps = self.patch_size
        # resize full image to a fixed patch size for quantum encoder
        if H == ps and W == ps:
            patch = image
        else:
            patch = F.interpolate(image, size=(ps, ps), mode="bilinear", align_corners=False)
        return self.enc(patch)  # [B, n_qubits]


class ConcatFusion(nn.Module):
    """
    concat fusion: [cnn_vec, q_proj] -> fused
    - project q_feat to head_dim to keep fusion balanced
    """
    def __init__(
        self,
        cnn_dim: int,
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
            nn.Linear(q_hidden, cnn_dim),
        )
        self.use_gate = use_gate
        if self.use_gate:
            self.gate = nn.Sequential(
                nn.Linear(cnn_dim, gate_hidden),
                nn.ReLU(inplace=True),
                nn.Linear(gate_hidden, 1),
            )
            nn.init.constant_(self.gate[-1].bias, init_gate_bias)

    @property
    def out_dim(self):
        return None  # determined by caller (2 * cnn_dim)

    def forward(self, cnn_vec: torch.Tensor, q_feat: torch.Tensor):
        # cnn_vec: [B, cnn_dim], q_feat: [B, q_dim]
        q_proj = self.q_to_cnn(q_feat)        # [B, cnn_dim]
        if self.use_gate:
            g = torch.sigmoid(self.gate(cnn_vec))     # [B, 1]
            q_proj = g * q_proj
        fused = torch.cat([cnn_vec, q_proj], dim=1)  # [B, 2*cnn_dim]
        return fused


# =========================================================
# 3) HybridModel (TorchQuantum backend)
# =========================================================
class HybridModel_TQ(nn.Module):
    """
    CNN branch: image -> CNN(+ASPP) -> pool -> cnn_vec
    Quantum branch: image -> (center patch encoder) -> n_qubits -> VQC -> q_feat
    Fusion: concat(cnn_vec, gate*proj(q_feat)) -> classifier
    """
    def __init__(
        self,
        image_channels=3,
        cnn_channels=32,
        num_classes=8,

        # ASPP
        use_aspp=False,
        aspp_rates=(2, 4, 6),
        aspp_dropout=0.1,

        # Quantum
        use_quantum=True,
        n_qubits=8,
        q_layers=2,
        q_entangle="ring",
        measure_z=True,
        measure_zz=True,
        measure_xx=True,
        correlator_pairs="ring",
        q_reupload=True,
        q_input_scale=math.pi,
        q_use_readout=True,

        # Image->Quantum
        q_patch_size=32,        # use a fixed-size patch for the quantum branch
        q_enc_hidden=128,

        # Fusion / head
        head_dim=256,
        q_hidden=256,
        fusion_gate=True,
        gate_hidden=128,
        init_gate_bias=-1.0,
        dropout=0.2,
    ):
        super().__init__()
        self.use_aspp = use_aspp
        self.use_quantum = use_quantum

        # CNN backbone (identical to v5)
        self.image_cnn = nn.Sequential(
            nn.Conv2d(image_channels, cnn_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(cnn_channels, cnn_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(
            in_channels=cnn_channels,
            out_channels=cnn_channels,
            atrous_rates=aspp_rates,
            dropout=aspp_dropout,
        ) if self.use_aspp else nn.Identity()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flat = nn.Flatten()

        # Strong CNN representation
        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_channels, head_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        # Quantum branch (image -> qubits -> quantum)
        if self.use_quantum:
            self.img2q = ImageToQubits(
                in_channels=image_channels,
                patch_size=q_patch_size,
                n_qubits=n_qubits,
                hidden=q_enc_hidden,
            )
            self.quantum = QuantumLayerTorchQuantumV5(
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
            self.fusion = ConcatFusion(
                cnn_dim=head_dim,
                q_dim=self.quantum.out_dim,
                q_hidden=q_hidden,
                use_gate=fusion_gate,
                gate_hidden=gate_hidden,
                init_gate_bias=init_gate_bias,
            )
            fusion_dim = head_dim * 2
        else:
            self.img2q = None
            self.quantum = None
            self.fusion = None
            fusion_dim = head_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, head_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_dim, num_classes),
        )

    def forward(self, image):
        # CNN branch
        fmap = self.image_cnn(image)
        fmap = self.aspp(fmap)
        vec = self.flat(self.pool(fmap))     # [B, cnn_channels]
        cnn_vec = self.cnn_proj(vec)         # [B, head_dim]

        # Quantum branch (directly from image patch)
        if self.use_quantum:
            q_in = self.img2q(image)         # [B, n_qubits]
            q_feat = self.quantum(q_in)      # [B, q_out_dim]
            fused = self.fusion(cnn_vec, q_feat)   # [B, 2*head_dim]
            return self.classifier(fused)

        return self.classifier(cnn_vec)
