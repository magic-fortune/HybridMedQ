import math
import warnings

import torch
import torch.nn as nn

try:
    import torchquantum as tq
    import torchquantum.functional as tqf
    from torchquantum.measurement import expval_joint_analytical
except Exception as exc:
    tq = None
    tqf = None
    expval_joint_analytical = None
    _TQ_IMPORT_ERROR = exc
    _TQ_AVAILABLE = False
else:
    _TQ_IMPORT_ERROR = None
    _TQ_AVAILABLE = True


class QuantumLayerOptional(nn.Module):
    """TorchQuantum layer with a differentiable fallback when tq is unavailable."""

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
        self._tq_available = _TQ_AVAILABLE
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
        self.input_scale = nn.Parameter(torch.full((n_qubits,), float(input_scaling_init), dtype=torch.float32))
        self.reupload_scale = nn.Parameter(0.1 * torch.randn(q_layers, n_qubits, dtype=torch.float32))
        self.reupload_shift = nn.Parameter(torch.zeros(q_layers, n_qubits, dtype=torch.float32))
        self.readout_weights = nn.Parameter(0.01 * torch.randn(n_qubits, 3, dtype=torch.float32))

        self.pairs = self._build_pairs()
        if self._tq_available:
            self.measure_all = tq.MeasureAll(tq.PauliZ)
        else:
            self.measure_all = None
            warnings.warn(
                f"TorchQuantum import failed ({_TQ_IMPORT_ERROR}). "
                "Using differentiable surrogate quantum layer.",
                RuntimeWarning,
            )

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

    def _forward_fallback(self, x: torch.Tensor) -> torch.Tensor:
        state = x
        for l in range(self.q_layers):
            if self.reupload:
                state = state + torch.sin(x * self.reupload_scale[l] + self.reupload_shift[l])

            wx = self.weights[l]
            a = wx[:, 0].unsqueeze(0)
            b = wx[:, 1].unsqueeze(0)
            c = wx[:, 2].unsqueeze(0)
            state = torch.sin(state + a) + torch.cos(state * (1.0 + b)) + c * torch.tanh(state)

        if self.use_readout:
            r = self.readout_weights
            state = torch.tanh(state * (1.0 + r[:, 0].unsqueeze(0)) + r[:, 1].unsqueeze(0))
            state = state + r[:, 2].unsqueeze(0)

        obs = []
        z_all = torch.tanh(state)
        if self.measure_z:
            for i in range(self.n_qubits):
                obs.append(z_all[:, i])

        if self.measure_zz:
            for (i, j) in self.pairs:
                obs.append(z_all[:, i] * z_all[:, j])

        if self.measure_xx:
            x_all = torch.sin(state)
            for (i, j) in self.pairs:
                obs.append(x_all[:, i] * x_all[:, j])

        return torch.stack(obs, dim=1).to(dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.to(dtype=torch.float32)

        if not self._tq_available:
            return self._forward_fallback(x)

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

