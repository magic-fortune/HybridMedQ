from datetime import datetime
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm
import argparse
import csv

import pennylane as qml

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))





from preprocessing.data_loader import MultiDataset


#     dataset="medmnist:bloodmnist",
#     metadata_csv="data/metadata.csv",
#     image_size=,
#     batch_size=4,
#     epochs=50,
#     lr=2e-5,
#     n_qubits=4,
#     num_classes=2):
        # image_channels=3,
        # cnn_channels=16,

def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="medmnist:bloodmnist",
                   help="openi or medmnist:<name> e.g. medmnist:bloodmnist")
    p.add_argument("--mode", type=str, default='tq', help="number of cnn channels")
    p.add_argument("--image_size", type=tuple, default=(128, 128), help="resize to this")
    p.add_argument("--image_size_3d", type=int, default=64, help="resize 3D volume to this")
    p.add_argument(
        "--medmnist_size",
        type=int,
        default=64,
        help="pre-resize MedMNIST size (28/64/128/224); use 0 for None",
    )
    p.add_argument("--image_channels", type=int, default=3, help="number of image channels")
    p.add_argument("--cnn_channels", type=int, default=16, help="number of cnn channels")
    
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers (auto fallback to 0 if unsupported)")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay")
    p.add_argument("--grad_clip", type=float, default=1.0, help="clip grad norm; <=0 disables")
    p.add_argument("--lr_scheduler", type=str, default="plateau",
                   help="none|step|cosine|plateau")
    p.add_argument("--lr_step_size", type=int, default=10, help="for step scheduler")
    p.add_argument("--lr_gamma", type=float, default=0.5, help="for step/plateau scheduler")
    p.add_argument("--min_lr", type=float, default=1e-6, help="for cosine/plateau scheduler")
    p.add_argument("--n_qubits", type=int, default=8)
    p.add_argument("--num_classes", type=int, default=8, help="auto infer if None")
    p.add_argument("--use_quantum", type=int, default=1, help="1 to enable quantum branch, 0 for pure CNN head")
    p.add_argument("--class_weighted", type=int, default=0, help="1 to use inverse-frequency class weights")

    p.add_argument("--use_noise_adapter", type=bool,default=True)
    p.add_argument("--use_aspp",  type=bool,default=True)
    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--cuda_device", type=int, default=0, help="CUDA device index")
    p.add_argument("--save_path", type=str, default='outs/tmp')
    p.add_argument("--resume_path", type=str, default="", help="path to model state_dict to resume")
    p.add_argument("--start_epoch", type=int, default=1, help="start epoch index (1-based)")
    p.add_argument("--max_train_batches", type=int, default=0, help="limit train batches per epoch (0=all)")
    p.add_argument("--max_val_batches", type=int, default=0, help="limit val batches per epoch (0=all)")
    p.add_argument("--val_interval", type=int, default=200, help="run validation every N train batches (0=only each epoch)")
    p.add_argument("--no_save", type=int, default=0, help="skip saving metrics/plots/models")
    p.add_argument("--return_metrics", type=int, default=0, help="return metrics from train()")
    p.add_argument("--report_f1", type=int, default=0, help="1 to print per-epoch val precision/recall/F1")
    p.add_argument("--strict_quantum", type=int, default=1, help="1 to disallow compat fallback for tq")

    #  quantum hyperparams (for tuning)
    p.add_argument("--q_layers", type=int, default=2)
    p.add_argument("--q_entangle", type=str, default="full")
    p.add_argument("--measure_z", type=int, default=1)
    p.add_argument("--measure_zz", type=int, default=1)
    p.add_argument("--measure_xx", type=int, default=0)
    p.add_argument("--correlator_pairs", type=str, default="ring")
    p.add_argument("--q_reupload", type=int, default=1)
    p.add_argument("--q_input_scale", type=float, default=math.pi)
    p.add_argument("--q_use_readout", type=int, default=1)
    p.add_argument("--q_patch_size", type=int, default=32)
    p.add_argument("--q_enc_hidden", type=int, default=128)
    p.add_argument("--q_hidden", type=int, default=256)
    p.add_argument("--head_dim", type=int, default=256)
    p.add_argument("--fusion_gate", type=int, default=1)
    p.add_argument("--gate_hidden", type=int, default=128)
    p.add_argument("--init_gate_bias", type=float, default=-2.0)
    p.add_argument("--dropout", type=float, default=0.2)

    return p



# -------------------------
# random seed
# -------------------------

import random
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_num_workers(requested: int) -> int:
    if requested <= 0:
        return 0
    try:
        import multiprocessing as mp
        lock = mp.Lock()
        del lock
        return requested
    except Exception as exc:
        print(f"[WARN] DataLoader multiprocessing unavailable ({exc}); fallback to num_workers=0")
        return 0


def _build_class_weights(dataset, num_classes: int):
    counts = torch.zeros(num_classes, dtype=torch.float32)
    for _, label in dataset:
        idx = int(label)
        if 0 <= idx < num_classes:
            counts[idx] += 1.0
    counts = torch.clamp(counts, min=1.0)
    weights = counts.sum() / (counts * float(num_classes))
    return weights


def _resolve_device(cuda_device: int) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    device_count = torch.cuda.device_count()
    if device_count <= 0:
        return torch.device("cpu")
    if cuda_device < 0 or cuda_device >= device_count:
        print(
            f"[WARN] cuda_device={cuda_device} is invalid (available: 0..{device_count - 1}); fallback to cuda:0"
        )
        return torch.device("cuda:0")
    return torch.device(f"cuda:{cuda_device}")


def _load_state_dict_file(path: str, device: torch.device):
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        return state["state_dict"]
    return state


def _compute_macro_prf1(preds: torch.Tensor, labels: torch.Tensor, num_classes: int):
    if preds.numel() == 0 or labels.numel() == 0:
        return 0.0, 0.0, 0.0

    preds = preds.to(dtype=torch.long)
    labels = labels.to(dtype=torch.long)
    k = max(int(num_classes), 1)

    cm = torch.bincount(labels * k + preds, minlength=k * k).reshape(k, k).to(dtype=torch.float32)
    tp = cm.diag()
    fp = cm.sum(dim=0) - tp
    fn = cm.sum(dim=1) - tp

    eps = 1e-12
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)

    support_mask = cm.sum(dim=1) > 0
    if support_mask.any():
        precision = precision[support_mask].mean().item()
        recall = recall[support_mask].mean().item()
        f1 = f1[support_mask].mean().item()
    else:
        precision = 0.0
        recall = 0.0
        f1 = 0.0
    return precision, recall, f1




# -------------------------
# logging
# -------------------------
def _to_float_or_empty(x):
    if x is None or x == "":
        return ""
    return float(x)


def log_metrics(
    epoch,
    loss,
    acc,
    val_loss,
    val_acc,
    save_path,
    precision="",
    recall="",
    f1="",
    split="",
):
    csv_path = os.path.join(save_path, "metrics.csv")
    row = {
        "epoch": epoch,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "loss": _to_float_or_empty(loss),
        "accuracy": _to_float_or_empty(acc),
        "val_loss": _to_float_or_empty(val_loss),
        "val_accuracy": _to_float_or_empty(val_acc),
        "precision": _to_float_or_empty(precision),
        "recall": _to_float_or_empty(recall),
        "f1": _to_float_or_empty(f1),
        "split": split if split is not None else "",
    }

    fieldnames = [
        "epoch",
        "time",
        "loss",
        "accuracy",
        "val_loss",
        "val_accuracy",
        "precision",
        "recall",
        "f1",
        "split",
    ]

    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode="a" if file_exists else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


# -------------------------
# plotting
# -------------------------
def save_accuracy_plot(train_acc_list, val_acc_list, save_path, filename=None):
    plt.figure()
    plt.plot(train_acc_list, label="Training Accuracy")
    plt.plot(val_acc_list, label="Validation Accuracy")
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend()
    if filename is None:
        filename = os.path.join(save_path, "accuracy_plots", f"accuracy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filename, dpi=200)
    plt.close()

def save_loss_plot(train_loss_list, val_loss_list, save_path, filename=None):
    plt.figure()
    plt.plot(train_loss_list, label="Training Loss")
    plt.plot(val_loss_list, label="Validation Loss")
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend()
    if filename is None:
        filename = os.path.join(save_path, "accuracy_plots", f"loss_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(filename, dpi=200)
    plt.close()

def _save_line_plot(x, y, xlabel, ylabel, title, save_path, filename):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(filename, dpi=200)
    plt.close()


def _save_two_line_plot(x1, y1, x2, y2, xlabel, ylabel, title, label1, label2, save_path, filename):
    plt.figure()
    plt.plot(x1, y1, label=label1)
    plt.plot(x2, y2, label=label2)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig(filename, dpi=200)
    plt.close()


# -------------------------
# quantum circuit visualization
# -------------------------
def save_quantum_circuit_diagram(model: HybridQuantumMultimodalModel, save_path):
    """
    Save the quantum circuit figure for the model's quantum layer.
    """
    qml.drawer.use_style("black_white")

    # We draw using a sample input of size n_qubits
    n_qubits = model.n_qubits
    sample_x = torch.full((n_qubits,), 0.1, dtype=torch.float32)
    sample_w = model.quantum.weights.detach().cpu()

    # qml.draw_mpl expects a qnode; in our model we stored it as model.quantum.circuit
    fig, ax = qml.draw_mpl(model.quantum.circuit)(sample_x, sample_w)
    out_path = os.path.join(save_path, "quantum_circuits", f"circuit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# -------------------------
# eval helper
# -------------------------
@torch.no_grad()
def evaluate(model, loader, device, criterion, max_batches: int = 0, report_f1: bool = False):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    all_preds = []
    all_labels = []
    inferred_num_classes = None

    for batch_idx, (img, labels) in enumerate(loader):
        img = img.to(device)
        labels = labels.to(device)

        logits = model(img)
        if inferred_num_classes is None:
            inferred_num_classes = int(logits.shape[1]) if logits.dim() > 1 else 1
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_n += labels.size(0)
        if report_f1:
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())
        if max_batches and (batch_idx + 1) >= max_batches:
            break

    avg_loss = total_loss / max(total_n, 1)
    avg_acc = total_correct / max(total_n, 1)
    if report_f1:
        preds_cat = torch.cat(all_preds, dim=0) if all_preds else torch.empty(0, dtype=torch.long)
        labels_cat = torch.cat(all_labels, dim=0) if all_labels else torch.empty(0, dtype=torch.long)
        precision, recall, f1 = _compute_macro_prf1(
            preds_cat,
            labels_cat,
            num_classes=(inferred_num_classes or 1),
        )
        return avg_loss, avg_acc, precision, recall, f1
    return avg_loss, avg_acc


# -------------------------
# train
# -------------------------
# def train(
#     dataset="medmnist:bloodmnist",
#     metadata_csv="data/metadata.csv",
#     image_size=(128, 128),
#     batch_size=4,
#     epochs=50,
#     lr=2e-5,
#     n_qubits=4,
#     num_classes=2):

def train(args):
    set_seed(args.seed)

    # normalize save_path: ensure it includes dataset, resized image size, and mode after "out"
    def _sanitize_dataset(name: str) -> str:
        return name.replace(":", "_").replace("/", "_")

    def _format_image_size(size) -> str:
        if isinstance(size, str):
            s = size.strip().strip("()[]")
            if "," in s:
                parts = [p.strip() for p in s.split(",") if p.strip()]
                if len(parts) >= 2:
                    return f"{parts[0]}x{parts[1]}"
                if len(parts) == 1:
                    return f"{parts[0]}x{parts[0]}"
            if s:
                return f"{s}x{s}"
            return "unknown"
        if isinstance(size, (tuple, list)) and len(size) >= 2:
            return f"{size[0]}x{size[1]}"
        if isinstance(size, int):
            return f"{size}x{size}"
        return str(size)

    def _infer_resized_size(ds) -> str:
        try:
            sample = ds[0][0]
            if isinstance(sample, torch.Tensor) and sample.dim() >= 2:
                h, w = int(sample.shape[-2]), int(sample.shape[-1])
                return f"{h}x{w}"
        except Exception:
            pass
        return _format_image_size(args.image_size)

    base_save = os.path.normpath(args.save_path)
    parts = base_save.split(os.sep)
    if parts and parts[0] == "outs_final_en":
        base_root = "outs_final_en"
    else:
        base_root = "outs_final_en"
    dataset_tag = _sanitize_dataset(args.dataset)
    mode_tag = str(args.mode)

    
    device = _resolve_device(args.cuda_device)
    print("device:", device)
    if args.dataset == 'openi':
        dataset = MultiDataset(
            dataset=args.dataset,
            split="train",
            root="data",
            metadata_path="data/metadata.csv",
            image_size=args.image_size,
        )
        
                # simple split: 80/20
        n_total = len(dataset)
        n_val = max(1, int(0.2 * n_total))
        n_train = n_total - n_val
        train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])
    elif 'medmnist' in args.dataset:
        ds_name = args.dataset.split(':')[-1].lower()
        is_3d = ds_name.endswith("3d")
        medmnist_size = None if args.medmnist_size == 0 else args.medmnist_size
        if medmnist_size is not None and not is_3d:
            args.image_size = (medmnist_size, medmnist_size)
        if medmnist_size is not None and is_3d:
            args.image_size_3d = medmnist_size
        train_set = MultiDataset(
        dataset=args.dataset,
        split="train",
        root="data",
        source="npz",
        medmnist_size=medmnist_size,
        image_size=args.image_size,
        image_size_3d=args.image_size_3d if is_3d else None,
    )

        val_set = MultiDataset( 
            dataset=args.dataset,
            split="val",
            root="data",
            source="npz",
            medmnist_size=medmnist_size,
            image_size=args.image_size,
            image_size_3d=args.image_size_3d if is_3d else None,
        )

    loader_workers = _resolve_num_workers(args.num_workers)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=loader_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=loader_workers)

    if 'medmnist' in args.dataset:
        size_val = args.medmnist_size
        size_tag = f"img{size_val}" if size_val != 0 else "imgNone"
    else:
        size_tag = f"img{_infer_resized_size(train_set)}"
    args.save_path = os.path.join(base_root, dataset_tag, size_tag, mode_tag)
    if args.mode in {"v5", "v5_tq", "v5_tq_3d"}:
        args.save_path = os.path.join(args.save_path, f"q{int(args.n_qubits)}")

    if not args.no_save:
        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(os.path.join(args.save_path, "accuracy_plots"), exist_ok=True)
        os.makedirs(os.path.join(args.save_path, "quantum_circuits"), exist_ok=True)
        os.makedirs(os.path.join(args.save_path, "checkpoints"), exist_ok=True)



    if args.mode == 'tq':
        try:
            from quantum.hybrid_model_torch_tq import HybridModel_TQ
            tq_backend = "torchquantum"
        except Exception as exc:
            if int(args.strict_quantum):
                raise RuntimeError(
                    "Strict quantum mode is enabled, but torchquantum backend import failed. "
                    "Please install/repair qiskit dependencies or set --strict_quantum 0 to allow compat fallback."
                ) from exc
            print(f"[WARN] Failed to import tq torchquantum backend ({exc}); using compat backend.")
            from quantum.hybrid_model_torch_tq_compat import HybridModel_TQ
            tq_backend = "compat"
        print(f"[INFO] tq backend: {tq_backend}")
        model = HybridModel_TQ(
            image_channels=args.image_channels,
            cnn_channels=32,
            num_classes=args.num_classes,
            use_aspp=False,
            use_quantum=True,

            n_qubits=args.n_qubits,
            q_layers=args.q_layers,
            q_entangle=args.q_entangle,
            measure_z=bool(args.measure_z),
            measure_zz=bool(args.measure_zz),
            measure_xx=bool(args.measure_xx),
            correlator_pairs=args.correlator_pairs,
            q_reupload=bool(args.q_reupload),
            q_input_scale=args.q_input_scale,
            q_use_readout=bool(args.q_use_readout),
            q_patch_size=args.q_patch_size,
            q_enc_hidden=args.q_enc_hidden,
            head_dim=args.head_dim,
            q_hidden=args.q_hidden,
            fusion_gate=bool(args.fusion_gate),
            gate_hidden=args.gate_hidden,
            init_gate_bias=args.init_gate_bias,
            dropout=args.dropout,
        ).to(device)

    elif args.mode == 'tq_3d':
        from quantum.hybrid_model_torch_tq_3d import HybridModel_TQ_3D
        model = HybridModel_TQ_3D(
            image_channels=args.image_channels,
            cnn_channels=args.cnn_channels,
            num_classes=args.num_classes,
            use_quantum=bool(args.use_quantum),

            n_qubits=args.n_qubits,
            q_layers=args.q_layers,
            q_entangle=args.q_entangle,
            measure_z=bool(args.measure_z),
            measure_zz=bool(args.measure_zz),
            measure_xx=bool(args.measure_xx),
            correlator_pairs=args.correlator_pairs,
            q_reupload=bool(args.q_reupload),
            q_input_scale=args.q_input_scale,
            q_use_readout=bool(args.q_use_readout),
            q_patch_size=args.q_patch_size,
            q_enc_hidden=args.q_enc_hidden,
            head_dim=args.head_dim,
            q_hidden=args.q_hidden,
            fusion_gate=bool(args.fusion_gate),
            gate_hidden=args.gate_hidden,
            init_gate_bias=args.init_gate_bias,
            dropout=args.dropout,
        ).to(device)
    
    
    # Save initial quantum circuit diagram
    # save_quantum_circuit_diagram(model)

    # optional resume
    if args.resume_path:
        if os.path.exists(args.resume_path):
            try:
                state = _load_state_dict_file(args.resume_path, device)
                model.load_state_dict(state)
                print(f"[INFO] Resumed from: {args.resume_path}")
            except Exception as exc:
                print(f"[WARN] Failed to resume from '{args.resume_path}': {exc}; train from scratch")
        else:
            print(f"[WARN] resume_path does not exist: {args.resume_path}; train from scratch")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma
        )
    elif args.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs), eta_min=args.min_lr
        )
    elif args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=args.lr_gamma, patience=2, min_lr=args.min_lr
        )
    else:
        scheduler = None
    if args.class_weighted:
        class_weights = _build_class_weights(train_set, int(args.num_classes)).to(device)
        print(f"class_weights: {class_weights.detach().cpu().tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    train_acc_steps = []
    train_loss_steps = []
    val_acc_steps = []
    val_loss_steps = []
    lr_steps = []
    step_indices = []
    val_step_indices = []
    epoch_indices = []
    train_acc_epochs = []
    val_acc_epochs = []
    train_loss_epochs = []
    val_loss_epochs = []
    lr_epochs = []
    best_val_acc = float("-inf")
    best_model_path = os.path.join(args.save_path, "best_model.pth")
    last_model_path = os.path.join(args.save_path, "last_model.pth")
    start_epoch = max(1, int(args.start_epoch))
    is_resume_run = bool(args.resume_path) or (start_epoch > 1)

    # metrics.csv handling:
    # - resume run: keep and recover historical best
    # - fresh run: reset old metrics.csv to avoid mixed runs in one file
    csv_path = os.path.join(args.save_path, "metrics.csv")
    if is_resume_run and os.path.exists(csv_path):
        try:
            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))
            if rows:
                best_val_acc = max(float(r["val_accuracy"]) for r in rows)
        except Exception:
            pass
    elif (not is_resume_run) and (not args.no_save) and os.path.exists(csv_path):
        try:
            os.remove(csv_path)
            print(f"[INFO] Fresh run: reset existing metrics file at {csv_path}")
        except Exception as exc:
            print(f"[WARN] Failed to reset old metrics file '{csv_path}': {exc}")

    val_acc = float("nan")
    global_step = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_n = 0

        total_batches = len(train_loader) if args.max_train_batches == 0 else min(len(train_loader), args.max_train_batches)
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{args.epochs}", leave=True, total=total_batches)
        for batch_idx, (img, labels) in enumerate(pbar):
            img = img.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            logits = model(img)
            loss = criterion(logits, labels)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

            epoch_loss += loss.item() * labels.size(0)
            preds = torch.argmax(logits, dim=1)
            epoch_correct += (preds == labels).sum().item()
            epoch_n += labels.size(0)

            batch_acc = (preds == labels).float().mean().item()
            current_lr = optimizer.param_groups[0]["lr"]
            global_step += 1
            step_indices.append(global_step)
            train_acc_steps.append(batch_acc)
            train_loss_steps.append(loss.item())
            lr_steps.append(current_lr)

            if args.val_interval and (global_step % args.val_interval == 0):
                v_loss, v_acc = evaluate(model, val_loader, device, criterion, max_batches=args.max_val_batches)
                val_step_indices.append(global_step)
                val_loss_steps.append(v_loss)
                val_acc_steps.append(v_acc)
            if args.max_train_batches and (batch_idx + 1) >= args.max_train_batches:
                break

        train_loss = epoch_loss / max(epoch_n, 1)
        train_acc = epoch_correct / max(epoch_n, 1)

        val_precision, val_recall, val_f1 = "", "", ""
        if args.report_f1:
            val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate(
                model,
                val_loader,
                device,
                criterion,
                max_batches=args.max_val_batches,
                report_f1=True,
            )
        else:
            val_loss, val_acc = evaluate(model, val_loader, device, criterion, max_batches=args.max_val_batches)

        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # always capture full-epoch validation point
        val_step_indices.append(global_step)
        val_loss_steps.append(val_loss)
        val_acc_steps.append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        if args.report_f1:
            print(
                f"Epoch {epoch}/{args.epochs} | lr={current_lr:.6g} | train_loss={train_loss:.4f} acc={train_acc:.4f} "
                f"| val_loss={val_loss:.4f} acc={val_acc:.4f} | val_precision={val_precision:.4f} val_recall={val_recall:.4f} val_f1={val_f1:.4f}"
            )
        else:
            print(f"Epoch {epoch}/{args.epochs} | lr={current_lr:.6g} | train_loss={train_loss:.4f} acc={train_acc:.4f} | val_loss={val_loss:.4f} acc={val_acc:.4f}")

        epoch_indices.append(epoch)
        train_acc_epochs.append(train_acc)
        val_acc_epochs.append(val_acc)
        train_loss_epochs.append(train_loss)
        val_loss_epochs.append(val_loss)
        lr_epochs.append(current_lr)

        if not args.no_save:
            log_metrics(
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                args.save_path,
                precision=val_precision,
                recall=val_recall,
                f1=val_f1,
                split="val",
            )
            plots_dir = os.path.join(args.save_path, "accuracy_plots")
            os.makedirs(plots_dir, exist_ok=True)
            # accuracy_plots: keep original single accuracy plot per epoch
            save_accuracy_plot(
                train_acc_epochs,
                val_acc_epochs,
                args.save_path,
                filename=os.path.join(plots_dir, f"accuracy_epoch_{epoch:03d}.png"),
            )
            save_loss_plot(
                train_loss_epochs,
                val_loss_epochs,
                args.save_path,
                filename=os.path.join(plots_dir, f"loss_epoch_{epoch:03d}.png"),
            )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            if not args.no_save:
                torch.save(model.state_dict(), best_model_path)

        # Save the latest model
        if not args.no_save:
            torch.save(model.state_dict(), last_model_path)
        
        if not args.no_save and (epoch % 5 == 0):
            ckpt_path = os.path.join(args.save_path, "checkpoints", f"checkpoint_epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), ckpt_path)


    # final plots already saved each epoch; keep final snapshot
    if not args.no_save and epoch_indices:
        plots_dir = os.path.join(args.save_path, "accuracy_plots")
        vis_dir = os.path.join(args.save_path, "vis")
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)
        save_accuracy_plot(
            train_acc_epochs,
            val_acc_epochs,
            args.save_path,
            filename=os.path.join(plots_dir, f"accuracy_epoch_{epoch_indices[-1]:03d}.png"),
        )
        save_loss_plot(
            train_loss_epochs,
            val_loss_epochs,
            args.save_path,
            filename=os.path.join(plots_dir, f"loss_epoch_{epoch_indices[-1]:03d}.png"),
        )
        # vis: keep only 6 final plots
        _save_line_plot(
            epoch_indices,
            train_acc_epochs,
            "Epoch",
            "Accuracy",
            "Training Accuracy",
            args.save_path,
            os.path.join(vis_dir, "train_accuracy.png"),
        )
        _save_line_plot(
            epoch_indices,
            val_acc_epochs,
            "Epoch",
            "Accuracy",
            "Validation Accuracy",
            args.save_path,
            os.path.join(vis_dir, "val_accuracy.png"),
        )
        _save_line_plot(
            epoch_indices,
            train_loss_epochs,
            "Epoch",
            "Loss",
            "Training Loss",
            args.save_path,
            os.path.join(vis_dir, "train_loss.png"),
        )
        _save_line_plot(
            epoch_indices,
            val_loss_epochs,
            "Epoch",
            "Loss",
            "Validation Loss",
            args.save_path,
            os.path.join(vis_dir, "val_loss.png"),
        )
        _save_two_line_plot(
            epoch_indices,
            train_acc_epochs,
            epoch_indices,
            val_acc_epochs,
            "Epoch",
            "Accuracy",
            "Training vs Validation Accuracy",
            "Training Accuracy",
            "Validation Accuracy",
            args.save_path,
            os.path.join(vis_dir, "train_val_accuracy.png"),
        )
        _save_two_line_plot(
            epoch_indices,
            train_loss_epochs,
            epoch_indices,
            val_loss_epochs,
            "Epoch",
            "Loss",
            "Training vs Validation Loss",
            "Training Loss",
            "Validation Loss",
            args.save_path,
            os.path.join(vis_dir, "train_val_loss.png"),
        )
        _save_line_plot(
            epoch_indices,
            lr_epochs,
            "Epoch",
            "Learning Rate",
            "Learning Rate Schedule",
            args.save_path,
            os.path.join(vis_dir, "learning_rate.png"),
        )
    elif not args.no_save:
        print("[WARN] No epoch executed; skip final plot generation")
    # optional test evaluation for medmnist/openi if test split exists
    if not args.no_save:
        test_loader = None
        if args.dataset == 'openi':
            test_loader = None
        elif 'medmnist' in args.dataset:
            ds_name = args.dataset.split(':')[-1].lower()
            is_3d = ds_name.endswith("3d")
            medmnist_size = None if args.medmnist_size == 0 else args.medmnist_size
            test_set = MultiDataset(
                dataset=args.dataset,
                split="test",
                root="data",
                source="npz",
                medmnist_size=medmnist_size,
                image_size=args.image_size,
                image_size_3d=args.image_size_3d if is_3d else None,
            )
            test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=loader_workers)

        if test_loader is not None:
            # best model
            if os.path.exists(best_model_path):
                try:
                    state = _load_state_dict_file(best_model_path, device)
                    model.load_state_dict(state)
                    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(
                        model,
                        test_loader,
                        device,
                        criterion,
                        max_batches=0,
                        report_f1=True,
                    )
                    log_metrics(
                        "test_best",
                        test_loss,
                        test_acc,
                        "",
                        "",
                        args.save_path,
                        precision=test_precision,
                        recall=test_recall,
                        f1=test_f1,
                        split="test",
                    )
                    print(
                        f"[TEST best] loss={test_loss:.4f} acc={test_acc:.4f} "
                        f"precision={test_precision:.4f} recall={test_recall:.4f} f1={test_f1:.4f}"
                    )
                except Exception as exc:
                    print(f"[WARN] test_best evaluation skipped (load/eval failed): {exc}")
            # last model
            if os.path.exists(last_model_path):
                try:
                    state = _load_state_dict_file(last_model_path, device)
                    model.load_state_dict(state)
                    test_loss, test_acc, test_precision, test_recall, test_f1 = evaluate(
                        model,
                        test_loader,
                        device,
                        criterion,
                        max_batches=0,
                        report_f1=True,
                    )
                    log_metrics(
                        "test_last",
                        test_loss,
                        test_acc,
                        "",
                        "",
                        args.save_path,
                        precision=test_precision,
                        recall=test_recall,
                        f1=test_f1,
                        split="test",
                    )
                    print(
                        f"[TEST last] loss={test_loss:.4f} acc={test_acc:.4f} "
                        f"precision={test_precision:.4f} recall={test_recall:.4f} f1={test_f1:.4f}"
                    )
                except Exception as exc:
                    print(f"[WARN] test_last evaluation skipped (load/eval failed): {exc}")

    if args.return_metrics:
        return model, {"best_val_acc": best_val_acc, "last_val_acc": val_acc}
    return model


if __name__ == "__main__":        
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
    parser = build_parser()
    args = parser.parse_args()
    train(args)
