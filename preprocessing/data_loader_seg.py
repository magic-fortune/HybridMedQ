import os
import random
from typing import List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def parse_exts(raw: str) -> Tuple[str, ...]:
    exts = []
    for ext in str(raw).split(","):
        ext = ext.strip().lower()
        if not ext:
            continue
        if not ext.startswith("."):
            ext = "." + ext
        exts.append(ext)
    return tuple(exts)


def _resolve_ph2_images_root(ph2_root: str) -> str:
    candidates = [
        ph2_root,
        os.path.join(ph2_root, "PH2 Dataset images"),
    ]
    for path in candidates:
        if os.path.isdir(path):
            has_case = any(
                os.path.isdir(os.path.join(path, name)) and name.upper().startswith("IMD")
                for name in os.listdir(path)
            )
            if has_case:
                return path
    raise FileNotFoundError(
        f"Cannot locate PH2 images root from '{ph2_root}'. "
        "Expected folders like IMDxxx under PH2 root."
    )


def _first_file(path_candidates: Sequence[str]) -> str:
    for path in path_candidates:
        if path and os.path.isfile(path):
            return path
    return ""


def _collect_case_paths(case_dir: str, case_id: str) -> Tuple[str, str]:
    image_candidates = [
        os.path.join(case_dir, f"{case_id}_Dermoscopic_Image", f"{case_id}.bmp"),
        os.path.join(case_dir, f"{case_id}_Dermoscopic_Image", f"{case_id}.png"),
        os.path.join(case_dir, "image", f"{case_id}.bmp"),
        os.path.join(case_dir, "image", f"{case_id}.png"),
    ]
    lesion_candidates = [
        os.path.join(case_dir, f"{case_id}_lesion", f"{case_id}_lesion.bmp"),
        os.path.join(case_dir, f"{case_id}_lesion", f"{case_id}_lesion.png"),
        os.path.join(case_dir, "lesion", f"{case_id}_lesion.bmp"),
        os.path.join(case_dir, "lesion", f"{case_id}_lesion.png"),
        os.path.join(case_dir, "lesion", f"{case_id}.bmp"),
        os.path.join(case_dir, "lesion", f"{case_id}.png"),
    ]

    if not any(os.path.isfile(p) for p in image_candidates):
        for sub in os.listdir(case_dir):
            sub_path = os.path.join(case_dir, sub)
            if not os.path.isdir(sub_path):
                continue
            sub_l = sub.lower()
            if "image" in sub_l:
                for f in sorted(os.listdir(sub_path)):
                    if os.path.splitext(f)[1].lower() in {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                        image_candidates.append(os.path.join(sub_path, f))
            if "lesion" in sub_l:
                for f in sorted(os.listdir(sub_path)):
                    if os.path.splitext(f)[1].lower() in {".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
                        lesion_candidates.append(os.path.join(sub_path, f))

    return _first_file(image_candidates), _first_file(lesion_candidates)


def collect_ph2_pairs(ph2_root: str) -> List[Tuple[str, str]]:
    images_root = _resolve_ph2_images_root(ph2_root)
    pairs: List[Tuple[str, str]] = []
    missing: List[str] = []

    case_ids = sorted(
        name
        for name in os.listdir(images_root)
        if os.path.isdir(os.path.join(images_root, name)) and name.upper().startswith("IMD")
    )

    for case_id in case_ids:
        case_dir = os.path.join(images_root, case_id)
        image_path, lesion_path = _collect_case_paths(case_dir, case_id)
        if image_path and lesion_path:
            pairs.append((image_path, lesion_path))
        else:
            missing.append(case_id)

    if missing:
        print(f"[PH2Loader] warning: {len(missing)} cases missing image/mask. Example: {missing[:5]}")
    if not pairs:
        raise ValueError(f"No valid image-mask pairs found under: {images_root}")

    return pairs


def split_pairs(
    pairs: Sequence[Tuple[str, str]],
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
):
    n = len(pairs)
    if n < 3:
        raise ValueError(f"PH2 requires at least 3 samples to build train/val/test splits, got {n}")

    val_split = float(val_split)
    test_split = float(test_split)
    if val_split < 0 or test_split < 0 or (val_split + test_split) >= 1.0:
        raise ValueError(
            f"Invalid split ratios: val_split={val_split}, test_split={test_split}; require val+test < 1"
        )

    n_val = max(1, int(round(n * val_split)))
    n_test = max(1, int(round(n * test_split)))
    n_train = n - n_val - n_test
    if n_train < 1:
        n_train = 1
        overflow = (n_val + n_test + n_train) - n
        while overflow > 0 and n_test > 1:
            n_test -= 1
            overflow -= 1
        while overflow > 0 and n_val > 1:
            n_val -= 1
            overflow -= 1

    random_gen = random.Random(seed)
    indices = list(range(n))
    random_gen.shuffle(indices)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:n_train + n_val + n_test]

    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]
    test_pairs = [pairs[i] for i in test_idx]
    return train_pairs, val_pairs, test_pairs


class PH2SegmentationDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[Tuple[str, str]],
        image_size: Tuple[int, int] = (256, 256),
        image_channels: int = 3,
        num_classes: int = 1,
        augment: bool = False,
    ):
        super().__init__()
        self.pairs = list(pairs)
        self.height, self.width = int(image_size[0]), int(image_size[1])
        self.image_channels = int(image_channels)
        self.num_classes = int(num_classes)
        self.augment = bool(augment)

        if not self.pairs:
            raise ValueError("PH2SegmentationDataset got empty pairs")

    def __len__(self):
        return len(self.pairs)

    def _apply_spatial_aug(self, image: Image.Image, mask: Image.Image):
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.2:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        k = random.randint(0, 3)
        if k:
            angle = 90 * k
            image = image.rotate(angle)
            mask = mask.rotate(angle)
        return image, mask

    def __getitem__(self, idx):
        image_path, mask_path = self.pairs[idx]

        image = Image.open(image_path)
        image = image.convert("L" if self.image_channels == 1 else "RGB")

        mask = Image.open(mask_path).convert("L")

        if self.augment:
            image, mask = self._apply_spatial_aug(image, mask)

        image = image.resize((self.width, self.height), resample=Image.BILINEAR)
        mask = mask.resize((self.width, self.height), resample=Image.NEAREST)

        image_np = np.asarray(image, dtype=np.float32)
        if image_np.ndim == 2:
            image_np = image_np[..., None]
        if self.image_channels == 3 and image_np.shape[-1] == 1:
            image_np = np.repeat(image_np, 3, axis=-1)
        if self.image_channels == 1 and image_np.shape[-1] != 1:
            image_np = image_np[..., :1]

        image_t = torch.from_numpy(image_np.transpose(2, 0, 1)) / 255.0

        mask_np = np.asarray(mask)
        if self.num_classes == 1:
            mask_np = (mask_np > 0).astype(np.float32)
            mask_t = torch.from_numpy(mask_np).unsqueeze(0)
        else:
            mask_t = torch.from_numpy(mask_np.astype(np.int64))

        return image_t, mask_t


def build_ph2_segmentation_sets(
    ph2_root: str,
    image_size: Tuple[int, int],
    image_channels: int,
    num_classes: int,
    val_split: float,
    test_split: float,
    seed: int,
    augment_train: bool = True,
):
    pairs = collect_ph2_pairs(ph2_root)
    train_pairs, val_pairs, test_pairs = split_pairs(
        pairs,
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )

    train_set = PH2SegmentationDataset(
        train_pairs,
        image_size=image_size,
        image_channels=image_channels,
        num_classes=num_classes,
        augment=augment_train,
    )
    val_set = PH2SegmentationDataset(
        val_pairs,
        image_size=image_size,
        image_channels=image_channels,
        num_classes=num_classes,
        augment=False,
    )
    test_set = PH2SegmentationDataset(
        test_pairs,
        image_size=image_size,
        image_channels=image_channels,
        num_classes=num_classes,
        augment=False,
    )

    print(
        f"[PH2Loader] split sizes -> train={len(train_set)}, val={len(val_set)}, test={len(test_set)}"
    )
    return train_set, val_set, test_set
