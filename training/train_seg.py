from datetime import datetime
import argparse
import csv
import math
import multiprocessing as mp
import os
import random
import sys
from typing import Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from scipy.spatial.distance import cdist
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocessing.data_loader_seg import build_ph2_segmentation_sets


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="ph2_seg", help="dataset tag used in save path")
    p.add_argument(
        "--mode",
        type=str,
        default="simple_cnn_seg_noq",
        help=(
            "tq_cnn_seg|tq_cnn_seg_noq|"
            "simple_cnn_seg|simple_cnn_seg_noq|"
            "simple_unet_seg|simple_unet_seg_noq|"
            "tq_seg|tq_seg_noq"
        ),
    )
    p.add_argument("--image_size", type=str, default="128,128", help="H,W or HxW")
    p.add_argument("--image_channels", type=int, default=3, help="number of input image channels")
    p.add_argument("--base_channels", type=int, default=32, help="base channels for segmentation model")
    p.add_argument("--num_classes", type=int, default=1, help="segmentation classes; use 1 for binary")

    p.add_argument("--train_images_dir", type=str, default="data/seg/train/images")
    p.add_argument("--train_masks_dir", type=str, default="data/seg/train/masks")
    p.add_argument("--val_images_dir", type=str, default="")
    p.add_argument("--val_masks_dir", type=str, default="")
    p.add_argument("--test_images_dir", type=str, default="")
    p.add_argument("--test_masks_dir", type=str, default="")
    p.add_argument("--val_split", type=float, default=0.2, help="used only when val dirs are not provided")
    p.add_argument("--test_split", type=float, default=0.15, help="used for PH2 auto split")
    p.add_argument("--image_exts", type=str, default=".png,.jpg,.jpeg,.bmp,.tif,.tiff")
    p.add_argument("--mask_exts", type=str, default=".png,.jpg,.jpeg,.bmp,.tif,.tiff")
    p.add_argument("--mask_suffix", type=str, default="", help="e.g. _mask for file_name_mask.png")
    p.add_argument("--ph2_root", type=str, default="PH2Dataset", help="PH2 dataset root folder")
    p.add_argument("--use_ph2_loader", type=int, default=1, help="1 to auto-load PH2 image/lesion pairs")
    p.add_argument("--augment_train", type=int, default=1, help="1 to enable train-time flips/rotations for PH2")

    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_scheduler", type=str, default="plateau", help="none|step|cosine|plateau")
    p.add_argument("--lr_step_size", type=int, default=10, help="for step scheduler")
    p.add_argument("--lr_gamma", type=float, default=0.5, help="for step/plateau scheduler")
    p.add_argument("--min_lr", type=float, default=1e-6, help="for cosine/plateau scheduler")
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--ignore_index", type=int, default=255, help="ignored label for multi-class segmentation")
    p.add_argument("--threshold", type=float, default=0.5, help="threshold for binary segmentation")
    p.add_argument("--bce_pos_weight", type=float, default=1.0, help="BCE positive class weight")

    p.add_argument("--seed", type=int, default=42, help="random seed")
    p.add_argument("--cuda_device", type=int, default=0, help="CUDA device index")
    p.add_argument("--save_path", type=str, default="outs/tmp_seg")
    p.add_argument("--resume_path", type=str, default="", help="path to model state_dict to resume")
    p.add_argument("--start_epoch", type=int, default=1, help="start epoch index (1-based)")
    p.add_argument("--max_train_batches", type=int, default=0, help="limit train batches per epoch (0=all)")
    p.add_argument("--max_val_batches", type=int, default=0, help="limit val batches per epoch (0=all)")
    p.add_argument("--val_interval", type=int, default=200, help="run validation every N train batches (0=only each epoch)")
    p.add_argument("--no_save", type=int, default=0, help="skip saving metrics/plots/models")
    p.add_argument("--return_metrics", type=int, default=0, help="return metrics from train()")

    p.add_argument("--use_aspp", type=int, default=1)
    p.add_argument("--use_quantum", type=int, default=1)

    p.add_argument("--n_qubits", type=int, default=8)
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
    p.add_argument("--fusion_gate", type=int, default=1)
    p.add_argument("--gate_hidden", type=int, default=128)
    p.add_argument("--init_gate_bias", type=float, default=-2.0)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--aspp_dropout", type=float, default=0.1)
    return p


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
        lock = mp.Lock()
        del lock
        return requested
    except Exception as exc:
        print(f"[WARN] DataLoader multiprocessing unavailable ({exc}); fallback to num_workers=0")
        return 0


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


def _parse_image_size(raw) -> Tuple[int, int]:
    if isinstance(raw, (tuple, list)) and len(raw) >= 2:
        return int(raw[0]), int(raw[1])
    if isinstance(raw, int):
        return int(raw), int(raw)
    s = str(raw).strip().lower().replace("x", ",").replace(" ", ",")
    parts = [p for p in s.split(",") if p]
    if len(parts) == 1:
        v = int(parts[0])
        return v, v
    if len(parts) >= 2:
        return int(parts[0]), int(parts[1])
    raise ValueError(f"Cannot parse image_size from: {raw}")


def _parse_exts(raw: str) -> Tuple[str, ...]:
    exts = []
    for ext in str(raw).split(","):
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        exts.append(ext)
    return tuple(exts)


class SegmentationDataset(Dataset):
    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        image_size: Tuple[int, int],
        image_channels: int,
        num_classes: int,
        image_exts: Sequence[str],
        mask_exts: Sequence[str],
        mask_suffix: str = "",
    ):
        super().__init__()
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.height, self.width = int(image_size[0]), int(image_size[1])
        self.image_channels = int(image_channels)
        self.num_classes = int(num_classes)
        self.image_exts = tuple(e.lower() for e in image_exts)
        self.mask_exts = tuple(e.lower() for e in mask_exts)
        self.mask_suffix = str(mask_suffix)

        if not os.path.isdir(self.images_dir):
            raise FileNotFoundError(f"images_dir not found: {self.images_dir}")
        if not os.path.isdir(self.masks_dir):
            raise FileNotFoundError(f"masks_dir not found: {self.masks_dir}")

        self.samples = self._build_samples()
        if not self.samples:
            raise ValueError(
                f"No image-mask pairs found in images_dir={self.images_dir}, masks_dir={self.masks_dir}."
            )

    def _build_samples(self):
        image_files = []
        for name in sorted(os.listdir(self.images_dir)):
            path = os.path.join(self.images_dir, name)
            if not os.path.isfile(path):
                continue
            if os.path.splitext(name)[1].lower() in self.image_exts:
                image_files.append(path)

        samples = []
        missing = []
        for img_path in image_files:
            mask_path = self._find_mask(img_path)
            if mask_path is None:
                missing.append(os.path.basename(img_path))
                continue
            samples.append((img_path, mask_path))

        if missing:
            print(f"[SegDataset] warning: {len(missing)} images have no matching mask. Example: {missing[:3]}")
        return samples

    def _find_mask(self, img_path: str):
        name = os.path.basename(img_path)
        stem, _ = os.path.splitext(name)
        candidates = []

        if self.mask_suffix:
            for ext in self.mask_exts:
                candidates.append(os.path.join(self.masks_dir, f"{stem}{self.mask_suffix}{ext}"))

        for ext in self.mask_exts:
            candidates.append(os.path.join(self.masks_dir, f"{stem}{ext}"))
        candidates.append(os.path.join(self.masks_dir, name))

        for p in candidates:
            if os.path.isfile(p):
                return p
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = Image.open(img_path)
        if self.image_channels == 1:
            image = image.convert("L")
        else:
            image = image.convert("RGB")
        image = image.resize((self.width, self.height), resample=Image.BILINEAR)
        image_np = np.asarray(image, dtype=np.float32)

        if image_np.ndim == 2:
            image_np = image_np[..., None]
        if self.image_channels == 1 and image_np.shape[-1] != 1:
            image_np = image_np[..., :1]
        if self.image_channels == 3 and image_np.shape[-1] == 1:
            image_np = np.repeat(image_np, 3, axis=-1)

        image_t = torch.from_numpy(image_np.transpose(2, 0, 1)) / 255.0

        mask = Image.open(mask_path)
        mask = mask.resize((self.width, self.height), resample=Image.NEAREST)
        mask_np = np.asarray(mask)
        if mask_np.ndim == 3:
            mask_np = mask_np[..., 0]

        if self.num_classes == 1:
            mask_np = (mask_np > 0).astype(np.float32)
            mask_t = torch.from_numpy(mask_np).unsqueeze(0)
        else:
            mask_t = torch.from_numpy(mask_np.astype(np.int64))

        return image_t, mask_t


def _to_float_or_empty(x):
    if x is None or x == "":
        return ""
    return float(x)


def log_metrics(
    epoch,
    loss,
    acc,
    dice,
    iou,
    hd95,
    jaccard,
    miou,
    val_loss,
    val_acc,
    val_dice,
    val_iou,
    val_hd95,
    val_jaccard,
    val_miou,
    save_path,
):
    csv_path = os.path.join(save_path, "metrics.csv")
    row = {
        "epoch": epoch,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "loss": _to_float_or_empty(loss),
        "accuracy": _to_float_or_empty(acc),
        "dice": _to_float_or_empty(dice),
        "iou": _to_float_or_empty(iou),
        "hd95": _to_float_or_empty(hd95),
        "jaccard": _to_float_or_empty(jaccard),
        "miou": _to_float_or_empty(miou),
        "val_loss": _to_float_or_empty(val_loss),
        "val_accuracy": _to_float_or_empty(val_acc),
        "val_dice": _to_float_or_empty(val_dice),
        "val_iou": _to_float_or_empty(val_iou),
        "val_hd95": _to_float_or_empty(val_hd95),
        "val_jaccard": _to_float_or_empty(val_jaccard),
        "val_miou": _to_float_or_empty(val_miou),
    }

    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode="a" if file_exists else "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_accuracy_plot(train_acc_list, val_acc_list, save_path, filename=None):
    plt.figure()
    plt.plot(train_acc_list, label="Training Accuracy")
    plt.plot(val_acc_list, label="Validation Accuracy")
    plt.title("Pixel Accuracy")
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


def save_metric_plot(train_values, val_values, metric_name, save_path, filename):
    plt.figure()
    plt.plot(train_values, label=f"Training {metric_name}")
    plt.plot(val_values, label=f"Validation {metric_name}")
    plt.title(metric_name)
    plt.ylabel(metric_name)
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig(filename, dpi=200)
    plt.close()


def _save_line_plot(x, y, xlabel, ylabel, title, filename):
    plt.figure()
    plt.plot(x, y)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.savefig(filename, dpi=200)
    plt.close()


def _save_two_line_plot(x1, y1, x2, y2, xlabel, ylabel, title, label1, label2, filename):
    plt.figure()
    plt.plot(x1, y1, label=label1)
    plt.plot(x2, y2, label=label2)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend()
    plt.savefig(filename, dpi=200)
    plt.close()


def compute_loss(criterion, logits, masks, num_classes: int):
    if num_classes == 1:
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        return criterion(logits, masks.float())
    if masks.dim() == 4 and masks.size(1) == 1:
        masks = masks[:, 0]
    return criterion(logits, masks.long())


def compute_batch_stats(logits, masks, num_classes: int, threshold: float = 0.5, ignore_index: int = 255):
    if num_classes == 1:
        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
        target = masks > 0.5
        preds = torch.sigmoid(logits) >= threshold
        valid = torch.ones_like(target, dtype=torch.bool)

        pixel_correct = ((preds == target) & valid).sum().item()
        pixel_total = valid.sum().item()

        inter = np.array([(preds & target & valid).sum().item()], dtype=np.float64)
        pred_area = np.array([(preds & valid).sum().item()], dtype=np.float64)
        target_area = np.array([(target & valid).sum().item()], dtype=np.float64)
        union = np.array([((preds | target) & valid).sum().item()], dtype=np.float64)
        return pixel_correct, pixel_total, inter, pred_area, target_area, union

    preds = torch.argmax(logits, dim=1)
    if masks.dim() == 4 and masks.size(1) == 1:
        masks = masks[:, 0]
    target = masks.long()

    if ignore_index >= 0:
        valid = target != ignore_index
    else:
        valid = torch.ones_like(target, dtype=torch.bool)

    pixel_correct = ((preds == target) & valid).sum().item()
    pixel_total = valid.sum().item()

    inter = np.zeros(num_classes, dtype=np.float64)
    pred_area = np.zeros(num_classes, dtype=np.float64)
    target_area = np.zeros(num_classes, dtype=np.float64)
    union = np.zeros(num_classes, dtype=np.float64)

    for cls_id in range(num_classes):
        pred_c = (preds == cls_id) & valid
        target_c = (target == cls_id) & valid
        inter[cls_id] = (pred_c & target_c).sum().item()
        pred_area[cls_id] = pred_c.sum().item()
        target_area[cls_id] = target_c.sum().item()
        union[cls_id] = (pred_c | target_c).sum().item()

    return pixel_correct, pixel_total, inter, pred_area, target_area, union


def finalize_metrics(pixel_correct, pixel_total, inter, pred_area, target_area, union):
    eps = 1e-7
    pixel_acc = float(pixel_correct / max(pixel_total, 1))
    valid = (pred_area + target_area) > 0
    if np.any(valid):
        dice = float(np.mean((2.0 * inter[valid] + eps) / (pred_area[valid] + target_area[valid] + eps)))
        iou = float(np.mean((inter[valid] + eps) / (union[valid] + eps)))
    else:
        dice = 1.0
        iou = 1.0
    return pixel_acc, dice, iou


def _compute_binary_hd95_from_masks(pred_mask: np.ndarray, target_mask: np.ndarray):
    pred = pred_mask.astype(bool)
    target = target_mask.astype(bool)

    if not pred.any() and not target.any():
        return 0.0
    if pred.any() != target.any():
        return np.nan

    pred_surface = pred ^ binary_erosion(pred)
    target_surface = target ^ binary_erosion(target)

    pred_pts = np.argwhere(pred_surface)
    target_pts = np.argwhere(target_surface)
    if pred_pts.shape[0] == 0:
        pred_pts = np.argwhere(pred)
    if target_pts.shape[0] == 0:
        target_pts = np.argwhere(target)
    if pred_pts.shape[0] == 0 or target_pts.shape[0] == 0:
        return np.nan

    dmat = cdist(pred_pts, target_pts)
    d_pred_to_target = dmat.min(axis=1)
    d_target_to_pred = dmat.min(axis=0)
    all_d = np.concatenate([d_pred_to_target, d_target_to_pred], axis=0)
    return float(np.percentile(all_d, 95))


def _decode_preds_targets(logits, masks, num_classes: int, threshold: float):
    if num_classes == 1:
        if masks.dim() == 4 and masks.size(1) == 1:
            target = (masks[:, 0] > 0.5)
        elif masks.dim() == 3:
            target = (masks > 0.5)
        else:
            target = (masks.squeeze(1) > 0.5)
        pred = torch.sigmoid(logits[:, 0]) >= threshold
        return pred, target

    pred = torch.argmax(logits, dim=1)
    if masks.dim() == 4 and masks.size(1) == 1:
        target = masks[:, 0].long()
    else:
        target = masks.long()
    return pred, target


def _update_hd95_accumulator(
    logits,
    masks,
    num_classes: int,
    threshold: float,
    ignore_index: int,
    hd95_sums,
    hd95_counts,
):
    pred, target = _decode_preds_targets(logits, masks, num_classes=num_classes, threshold=threshold)
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()

    if num_classes == 1:
        for b in range(pred_np.shape[0]):
            hd95 = _compute_binary_hd95_from_masks(pred_np[b], target_np[b])
            if np.isfinite(hd95):
                hd95_sums[0] += float(hd95)
                hd95_counts[0] += 1
        return

    for b in range(pred_np.shape[0]):
        t = target_np[b]
        p = pred_np[b]
        valid = np.ones_like(t, dtype=bool)
        if ignore_index >= 0:
            valid = t != ignore_index
        for cls_id in range(num_classes):
            pred_mask = (p == cls_id) & valid
            target_mask = (t == cls_id) & valid
            hd95 = _compute_binary_hd95_from_masks(pred_mask, target_mask)
            if np.isfinite(hd95):
                hd95_sums[cls_id] += float(hd95)
                hd95_counts[cls_id] += 1


def finalize_hd95(hd95_sums, hd95_counts):
    valid = hd95_counts > 0
    if not np.any(valid):
        return float("nan")
    per_class = hd95_sums[valid] / hd95_counts[valid]
    return float(np.mean(per_class))


@torch.no_grad()
def evaluate(model, loader, device, criterion, num_classes: int, threshold: float, ignore_index: int, max_batches: int = 0):
    model.eval()
    total_loss = 0.0
    total_items = 0

    c = 1 if num_classes == 1 else num_classes
    pixel_correct = 0.0
    pixel_total = 0.0
    inter = np.zeros(c, dtype=np.float64)
    pred_area = np.zeros(c, dtype=np.float64)
    target_area = np.zeros(c, dtype=np.float64)
    union = np.zeros(c, dtype=np.float64)
    hd95_sums = np.zeros(c, dtype=np.float64)
    hd95_counts = np.zeros(c, dtype=np.int64)

    for batch_idx, (img, masks) in enumerate(loader):
        img = img.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        logits = model(img)
        loss = compute_loss(criterion, logits, masks, num_classes=num_classes)

        bsz = img.size(0)
        total_loss += loss.item() * bsz
        total_items += bsz

        b_correct, b_total, b_inter, b_pred, b_target, b_union = compute_batch_stats(
            logits,
            masks,
            num_classes=num_classes,
            threshold=threshold,
            ignore_index=ignore_index,
        )
        pixel_correct += b_correct
        pixel_total += b_total
        inter += b_inter
        pred_area += b_pred
        target_area += b_target
        union += b_union
        _update_hd95_accumulator(
            logits,
            masks,
            num_classes=num_classes,
            threshold=threshold,
            ignore_index=ignore_index,
            hd95_sums=hd95_sums,
            hd95_counts=hd95_counts,
        )

        if max_batches and (batch_idx + 1) >= max_batches:
            break

    avg_loss = total_loss / max(total_items, 1)
    pixel_acc, dice, iou = finalize_metrics(pixel_correct, pixel_total, inter, pred_area, target_area, union)
    hd95 = finalize_hd95(hd95_sums, hd95_counts)
    jaccard = iou
    miou = iou
    return avg_loss, pixel_acc, dice, iou, hd95, jaccard, miou


def train(args):
    set_seed(args.seed)
    args.image_size = _parse_image_size(args.image_size)
    image_exts = _parse_exts(args.image_exts)
    mask_exts = _parse_exts(args.mask_exts)

    def _sanitize_dataset(name: str) -> str:
        return name.replace(":", "_").replace("/", "_")

    base_root = "outs_final"
    dataset_tag = _sanitize_dataset(args.dataset)
    mode_tag = str(args.mode)

    device = _resolve_device(args.cuda_device)
    print("device:", device)

    use_ph2_loader = bool(args.use_ph2_loader)
    test_set = None
    if use_ph2_loader:
        train_set, val_set, test_set = build_ph2_segmentation_sets(
            ph2_root=args.ph2_root,
            image_size=args.image_size,
            image_channels=args.image_channels,
            num_classes=args.num_classes,
            val_split=args.val_split,
            test_split=args.test_split,
            seed=args.seed,
            augment_train=bool(args.augment_train),
        )
    else:
        full_train_set = SegmentationDataset(
            images_dir=args.train_images_dir,
            masks_dir=args.train_masks_dir,
            image_size=args.image_size,
            image_channels=args.image_channels,
            num_classes=args.num_classes,
            image_exts=image_exts,
            mask_exts=mask_exts,
            mask_suffix=args.mask_suffix,
        )

        if args.val_images_dir and args.val_masks_dir:
            train_set = full_train_set
            val_set = SegmentationDataset(
                images_dir=args.val_images_dir,
                masks_dir=args.val_masks_dir,
                image_size=args.image_size,
                image_channels=args.image_channels,
                num_classes=args.num_classes,
                image_exts=image_exts,
                mask_exts=mask_exts,
                mask_suffix=args.mask_suffix,
            )
        else:
            n_total = len(full_train_set)
            if n_total < 2:
                raise ValueError(
                    "train set has fewer than 2 samples; please provide --val_images_dir and --val_masks_dir."
                )
            n_val = max(1, int(args.val_split * n_total))
            n_val = min(n_val, n_total - 1)
            n_train = n_total - n_val
            gen = torch.Generator().manual_seed(args.seed)
            train_set, val_set = random_split(full_train_set, [n_train, n_val], generator=gen)

    loader_workers = _resolve_num_workers(args.num_workers)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=loader_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=loader_workers,
        pin_memory=torch.cuda.is_available(),
    )

    size_tag = f"img{args.image_size[0]}x{args.image_size[1]}"
    args.save_path = os.path.join(base_root, dataset_tag, size_tag, mode_tag)

    if not args.no_save:
        os.makedirs(args.save_path, exist_ok=True)
        os.makedirs(os.path.join(args.save_path, "accuracy_plots"), exist_ok=True)
        os.makedirs(os.path.join(args.save_path, "checkpoints"), exist_ok=True)

    use_quantum = bool(args.use_quantum)
    model = None

    if args.mode in {"tq_cnn_seg", "tq_cnn_seg_noq"}:
        if args.mode.endswith("_noq"):
            use_quantum = False
        from quantum.hybrid_model_torch_tq_segcnn import HybridModel_TQ_SegCNN

        model = HybridModel_TQ_SegCNN(
            image_channels=args.image_channels,
            cnn_channels=args.base_channels,
            num_classes=args.num_classes,
            use_aspp=bool(args.use_aspp),
            aspp_rates=(2, 4, 6),
            aspp_dropout=args.aspp_dropout,
            use_quantum=use_quantum,
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
            q_hidden=args.q_hidden,
            fusion_gate=bool(args.fusion_gate),
            gate_hidden=args.gate_hidden,
            init_gate_bias=args.init_gate_bias,
            dropout=args.dropout,
        ).to(device)

    elif args.mode in {"simple_cnn_seg", "simple_cnn_seg_noq"}:
        if args.mode.endswith("_noq"):
            use_quantum = False
        from quantum.simple_cnn_seg import SimpleCNNSeg

        model = SimpleCNNSeg(
            image_channels=args.image_channels,
            base_channels=args.base_channels,
            num_classes=args.num_classes,
            use_quantum=use_quantum,
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
            q_hidden=args.q_hidden,
            fusion_gate=bool(args.fusion_gate),
            gate_hidden=args.gate_hidden,
            init_gate_bias=args.init_gate_bias,
            dropout=args.dropout,
            use_aspp=False,
        ).to(device)

    elif args.mode in {"simple_unet_seg", "simple_unet_seg_noq"}:
        if args.mode.endswith("_noq"):
            use_quantum = False
        from quantum.simple_unet_seg import SimpleUNetSeg

        model = SimpleUNetSeg(
            image_channels=args.image_channels,
            base_channels=args.base_channels,
            num_classes=args.num_classes,
            use_quantum=use_quantum,
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
            q_hidden=args.q_hidden,
            fusion_gate=bool(args.fusion_gate),
            gate_hidden=args.gate_hidden,
            init_gate_bias=args.init_gate_bias,
            dropout=args.dropout,
            use_aspp=False,
        ).to(device)

    elif args.mode in {"tq_seg", "tq_seg_noq"}:
        if args.mode == "tq_seg_noq":
            use_quantum = False

        try:
            from quantum.hybrid_model_torch_seg import HybridModel_TQ_Seg
        except Exception as exc:
            print(f"[WARN] Failed to import segmentation quantum model ({exc}); forcing no-quantum mode")
            use_quantum = False
            args.mode = "tq_seg_noq"
            from quantum.hybrid_model_torch_seg_noq import HybridModel_TQ_Seg

        model = HybridModel_TQ_Seg(
            image_channels=args.image_channels,
            base_channels=args.base_channels,
            num_classes=args.num_classes,
            use_aspp=bool(args.use_aspp),
            aspp_rates=(2, 4, 6),
            aspp_dropout=args.aspp_dropout,
            use_quantum=use_quantum,
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
            q_hidden=args.q_hidden,
            fusion_gate=bool(args.fusion_gate),
            gate_hidden=args.gate_hidden,
            init_gate_bias=args.init_gate_bias,
            dropout=args.dropout,
        ).to(device)
    else:
        raise ValueError(
            f"Unsupported mode='{args.mode}'. "
            "Use tq_cnn_seg|tq_cnn_seg_noq|simple_cnn_seg|simple_cnn_seg_noq|simple_unet_seg|simple_unet_seg_noq|tq_seg|tq_seg_noq"
        )

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

    if args.num_classes == 1:
        if args.bce_pos_weight > 0:
            pos_weight = torch.tensor([args.bce_pos_weight], dtype=torch.float32, device=device)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()
    else:
        ignore_index = args.ignore_index if args.ignore_index >= 0 else -100
        criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

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
    train_dice_epochs = []
    val_dice_epochs = []
    train_iou_epochs = []
    val_iou_epochs = []
    lr_epochs = []

    best_val_dice = float("-inf")
    best_model_path = os.path.join(args.save_path, "best_model.pth")
    last_model_path = os.path.join(args.save_path, "last_model.pth")
    start_epoch = max(1, int(args.start_epoch))
    is_resume_run = bool(args.resume_path) or (start_epoch > 1)
    csv_path = os.path.join(args.save_path, "metrics.csv")
    if is_resume_run and os.path.exists(csv_path):
        try:
            with open(csv_path, newline="") as f:
                rows = list(csv.DictReader(f))
            vals = []
            for r in rows:
                v = r.get("val_dice", "")
                if v not in ("", None):
                    vals.append(float(v))
            if vals:
                best_val_dice = max(vals)
        except Exception:
            pass

    global_step = 0
    val_dice = 0.0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_items = 0

        c = 1 if args.num_classes == 1 else args.num_classes
        epoch_pixel_correct = 0.0
        epoch_pixel_total = 0.0
        epoch_inter = np.zeros(c, dtype=np.float64)
        epoch_pred_area = np.zeros(c, dtype=np.float64)
        epoch_target_area = np.zeros(c, dtype=np.float64)
        epoch_union = np.zeros(c, dtype=np.float64)
        epoch_hd95_sums = np.zeros(c, dtype=np.float64)
        epoch_hd95_counts = np.zeros(c, dtype=np.int64)

        total_batches = len(train_loader) if args.max_train_batches == 0 else min(len(train_loader), args.max_train_batches)
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{args.epochs}", leave=True, total=total_batches)
        for batch_idx, (img, masks) in enumerate(pbar):
            img = img.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(img)
            loss = compute_loss(criterion, logits, masks, num_classes=args.num_classes)
            loss.backward()
            optimizer.step()

            bsz = img.size(0)
            epoch_loss += loss.item() * bsz
            epoch_items += bsz

            b_correct, b_total, b_inter, b_pred, b_target, b_union = compute_batch_stats(
                logits,
                masks,
                num_classes=args.num_classes,
                threshold=args.threshold,
                ignore_index=args.ignore_index,
            )
            epoch_pixel_correct += b_correct
            epoch_pixel_total += b_total
            epoch_inter += b_inter
            epoch_pred_area += b_pred
            epoch_target_area += b_target
            epoch_union += b_union
            _update_hd95_accumulator(
                logits,
                masks,
                num_classes=args.num_classes,
                threshold=args.threshold,
                ignore_index=args.ignore_index,
                hd95_sums=epoch_hd95_sums,
                hd95_counts=epoch_hd95_counts,
            )

            batch_acc = b_correct / max(b_total, 1)
            current_lr = optimizer.param_groups[0]["lr"]
            global_step += 1
            step_indices.append(global_step)
            train_acc_steps.append(batch_acc)
            train_loss_steps.append(loss.item())
            lr_steps.append(current_lr)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{batch_acc:.4f}", lr=f"{current_lr:.2e}")

            if args.val_interval and (global_step % args.val_interval == 0):
                v_loss, v_acc, _, _, _, _, _ = evaluate(
                    model,
                    val_loader,
                    device,
                    criterion,
                    num_classes=args.num_classes,
                    threshold=args.threshold,
                    ignore_index=args.ignore_index,
                    max_batches=args.max_val_batches,
                )
                val_step_indices.append(global_step)
                val_loss_steps.append(v_loss)
                val_acc_steps.append(v_acc)
                model.train()

            if args.max_train_batches and (batch_idx + 1) >= args.max_train_batches:
                break

        train_loss = epoch_loss / max(epoch_items, 1)
        train_acc, train_dice, train_iou = finalize_metrics(
            epoch_pixel_correct,
            epoch_pixel_total,
            epoch_inter,
            epoch_pred_area,
            epoch_target_area,
            epoch_union,
        )
        train_hd95 = finalize_hd95(epoch_hd95_sums, epoch_hd95_counts)
        train_jaccard = train_iou
        train_miou = train_iou

        val_loss, val_acc, val_dice, val_iou, val_hd95, val_jaccard, val_miou = evaluate(
            model,
            val_loader,
            device,
            criterion,
            num_classes=args.num_classes,
            threshold=args.threshold,
            ignore_index=args.ignore_index,
            max_batches=args.max_val_batches,
        )

        if scheduler is not None:
            if args.lr_scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        val_step_indices.append(global_step)
        val_loss_steps.append(val_loss)
        val_acc_steps.append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{args.epochs} | lr={current_lr:.6g} "
            f"| train_loss={train_loss:.4f} acc={train_acc:.4f} dice={train_dice:.4f} iou={train_iou:.4f} "
            f"hd95={train_hd95:.4f} jaccard={train_jaccard:.4f} miou={train_miou:.4f} "
            f"| val_loss={val_loss:.4f} acc={val_acc:.4f} dice={val_dice:.4f} iou={val_iou:.4f} "
            f"hd95={val_hd95:.4f} jaccard={val_jaccard:.4f} miou={val_miou:.4f}"
        )

        epoch_indices.append(epoch)
        train_acc_epochs.append(train_acc)
        val_acc_epochs.append(val_acc)
        train_loss_epochs.append(train_loss)
        val_loss_epochs.append(val_loss)
        train_dice_epochs.append(train_dice)
        val_dice_epochs.append(val_dice)
        train_iou_epochs.append(train_iou)
        val_iou_epochs.append(val_iou)
        lr_epochs.append(current_lr)

        if not args.no_save:
            log_metrics(
                epoch,
                train_loss,
                train_acc,
                train_dice,
                train_iou,
                train_hd95,
                train_jaccard,
                train_miou,
                val_loss,
                val_acc,
                val_dice,
                val_iou,
                val_hd95,
                val_jaccard,
                val_miou,
                args.save_path,
            )
            plots_dir = os.path.join(args.save_path, "accuracy_plots")
            os.makedirs(plots_dir, exist_ok=True)
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
            save_metric_plot(
                train_dice_epochs,
                val_dice_epochs,
                metric_name="Dice",
                save_path=args.save_path,
                filename=os.path.join(plots_dir, f"dice_epoch_{epoch:03d}.png"),
            )
            save_metric_plot(
                train_iou_epochs,
                val_iou_epochs,
                metric_name="IoU",
                save_path=args.save_path,
                filename=os.path.join(plots_dir, f"iou_epoch_{epoch:03d}.png"),
            )

        if val_dice > best_val_dice:
            best_val_dice = val_dice
            if not args.no_save:
                torch.save(model.state_dict(), best_model_path)

        if not args.no_save:
            torch.save(model.state_dict(), last_model_path)

        if not args.no_save and (epoch % 5 == 0):
            ckpt_path = os.path.join(args.save_path, "checkpoints", f"checkpoint_epoch_{epoch:03d}.pth")
            torch.save(model.state_dict(), ckpt_path)

    if not args.no_save and epoch_indices:
        plots_dir = os.path.join(args.save_path, "accuracy_plots")
        vis_dir = os.path.join(args.save_path, "vis")
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

        _save_line_plot(
            epoch_indices,
            train_acc_epochs,
            "Epoch",
            "Pixel Accuracy",
            "Training Pixel Accuracy",
            os.path.join(vis_dir, "train_accuracy.png"),
        )
        _save_line_plot(
            epoch_indices,
            val_acc_epochs,
            "Epoch",
            "Pixel Accuracy",
            "Validation Pixel Accuracy",
            os.path.join(vis_dir, "val_accuracy.png"),
        )
        _save_line_plot(
            epoch_indices,
            train_loss_epochs,
            "Epoch",
            "Loss",
            "Training Loss",
            os.path.join(vis_dir, "train_loss.png"),
        )
        _save_line_plot(
            epoch_indices,
            val_loss_epochs,
            "Epoch",
            "Loss",
            "Validation Loss",
            os.path.join(vis_dir, "val_loss.png"),
        )
        _save_two_line_plot(
            epoch_indices,
            train_dice_epochs,
            epoch_indices,
            val_dice_epochs,
            "Epoch",
            "Dice",
            "Training vs Validation Dice",
            "Training Dice",
            "Validation Dice",
            os.path.join(vis_dir, "train_val_dice.png"),
        )
        _save_two_line_plot(
            epoch_indices,
            train_iou_epochs,
            epoch_indices,
            val_iou_epochs,
            "Epoch",
            "IoU",
            "Training vs Validation IoU",
            "Training IoU",
            "Validation IoU",
            os.path.join(vis_dir, "train_val_iou.png"),
        )
        _save_line_plot(
            epoch_indices,
            lr_epochs,
            "Epoch",
            "Learning Rate",
            "Learning Rate Schedule",
            os.path.join(vis_dir, "learning_rate.png"),
        )

    test_loader = None
    if not args.no_save:
        if test_set is not None:
            test_loader = DataLoader(
                test_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=loader_workers,
                pin_memory=torch.cuda.is_available(),
            )
        elif args.test_images_dir and args.test_masks_dir:
            if os.path.isdir(args.test_images_dir) and os.path.isdir(args.test_masks_dir):
                test_set = SegmentationDataset(
                    images_dir=args.test_images_dir,
                    masks_dir=args.test_masks_dir,
                    image_size=args.image_size,
                    image_channels=args.image_channels,
                    num_classes=args.num_classes,
                    image_exts=image_exts,
                    mask_exts=mask_exts,
                    mask_suffix=args.mask_suffix,
                )
                test_loader = DataLoader(
                    test_set,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=loader_workers,
                    pin_memory=torch.cuda.is_available(),
                )

    if not args.no_save and test_loader is not None:
        if os.path.exists(best_model_path):
            try:
                state = _load_state_dict_file(best_model_path, device)
                model.load_state_dict(state)
                test_loss, test_acc, test_dice, test_iou, test_hd95, test_jaccard, test_miou = evaluate(
                    model,
                    test_loader,
                    device,
                    criterion,
                    num_classes=args.num_classes,
                    threshold=args.threshold,
                    ignore_index=args.ignore_index,
                    max_batches=0,
                )
                log_metrics(
                    "test_best",
                    test_loss,
                    test_acc,
                    test_dice,
                    test_iou,
                    test_hd95,
                    test_jaccard,
                    test_miou,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    args.save_path,
                )
            except Exception as exc:
                print(f"[WARN] test_best evaluation skipped (load/eval failed): {exc}")

        if os.path.exists(last_model_path):
            try:
                state = _load_state_dict_file(last_model_path, device)
                model.load_state_dict(state)
                test_loss, test_acc, test_dice, test_iou, test_hd95, test_jaccard, test_miou = evaluate(
                    model,
                    test_loader,
                    device,
                    criterion,
                    num_classes=args.num_classes,
                    threshold=args.threshold,
                    ignore_index=args.ignore_index,
                    max_batches=0,
                )
                log_metrics(
                    "test_last",
                    test_loss,
                    test_acc,
                    test_dice,
                    test_iou,
                    test_hd95,
                    test_jaccard,
                    test_miou,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    args.save_path,
                )
            except Exception as exc:
                print(f"[WARN] test_last evaluation skipped (load/eval failed): {exc}")

    if args.return_metrics:
        return model, {"best_val_dice": best_val_dice, "last_val_dice": val_dice}
    return model


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    train(args)
