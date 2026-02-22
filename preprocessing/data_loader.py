import os
from typing import Tuple, Optional, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
import torch.nn.functional as F

from .xml_parser import extract_text_from_xml


Split = Literal["train", "val", "test"]
Source = Literal["npz", "medmnist"]


class MultiDataset(Dataset):
    """
    Unified dataset wrapper for:
      1) OpenI: uses metadata.csv with image_path/report_path/label
      2) MedMNIST: uses local .npz (recommended) OR medmnist package

    Returns:
      - (image_tensor, label_tensor)
    """

    def __init__(
        self,
        dataset: str,                         # "openi" or "bloodmnist" / "pathmnist" ...
        split: Split = "train",
        root: str = "data",
        image_size: Tuple[int, int] = (128, 128),
        image_size_3d: int = 64,

        # ---- OpenI ----
        metadata_path: Optional[str] = None,  # default: {root}/metadata.csv
        use_text: bool = False,               # 如果以后你要返回 text，再扩展

        # ---- MedMNIST ----
        source: Source = "npz",               # "npz"(local) or "medmnist"(download)
        medmnist_size: Optional[int] = None,  # 28/64/128/224 or None
        npz_path: Optional[str] = None,       # explicit local file path
        download: bool = False,               # only for source="medmnist"
    ):
        super().__init__()
        self.dataset = dataset.lower()
        self.split = split
        self.root = root
        self.image_size = image_size
        self.image_size_3d = image_size_3d
        self.use_text = use_text

        # common transforms
        self.tf_gray = T.Compose([T.Resize(self.image_size), T.ToTensor()])
        self.tf_rgb = T.Compose([T.Resize(self.image_size), T.ToTensor()])

        if self.dataset == "openi":
            self.mode = "openi"
            self._init_openi(metadata_path)
        else:
            self.mode = "medmnist"
            self._init_medmnist(
                name=self.dataset,
                source=source,
                medmnist_size=medmnist_size,
                npz_path=npz_path,
                download=download,
            )

    # -----------------------------
    # init: OpenI
    # -----------------------------
    def _init_openi(self, metadata_path: Optional[str]):
        if metadata_path is None:
            metadata_path = os.path.join(self.root, "metadata.csv")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"OpenI metadata not found: {metadata_path}")

        self.metadata_path = metadata_path
        self.metadata = pd.read_csv(metadata_path)
        if len(self.metadata) == 0:
            raise ValueError(f"metadata.csv is empty: {metadata_path}")

    # -----------------------------
    # init: MedMNIST
    # -----------------------------
    def _guess_npz_path(self, name: str, size: Optional[int]) -> str:
        base = 'medmnist_data'
        # preferred: name_size.npz
        name = name.split(':')[-1]
        if size is not None:
            p = os.path.join(base, f"{name}_{size}.npz")
            if os.path.exists(p):
                return p
        # fallback: name.npz
        p = os.path.join(base, f"{name}.npz")
        if os.path.exists(p):
            return p
        # last resort: raise with helpful message
        raise FileNotFoundError(
            f"Cannot find local npz for '{name}'. Tried:\n"
            f"  - {os.path.join(base, f'{name}_{size}.npz') if size is not None else '(skip size)'}\n"
            f"  - {os.path.join(base, f'{name}.npz')}\n"
            f"Your screenshot shows files like: bloodmnist_64.npz under data/medmnist_data/"
        )

    def _init_medmnist(
        self,
        name: str,
        source: Source,
        medmnist_size: Optional[int],
        npz_path: Optional[str],
        download: bool,
    ):
        self.med_name = name
        self.is_3d = name.split(":")[-1].lower().endswith("3d")
        self.med_size = medmnist_size
        self.med_source = source

        if source == "npz":
            if npz_path is None:
                npz_path = self._guess_npz_path(name, medmnist_size)
            self.npz_path = npz_path
            self._load_medmnist_npz(npz_path, split=self.split)

        elif source == "medmnist":
            # optional path: {root}/medmnist
            med_root = os.path.join(self.root, "medmnist")
            self._load_medmnist_pkg(name=name, split=self.split, root=med_root, size=medmnist_size, download=download)

        else:
            raise ValueError(f"Unknown source={source}. Use 'npz' or 'medmnist'.")

    # -----------------------------
    # MedMNIST from NPZ
    # -----------------------------
    def _load_medmnist_npz(self, npz_path: str, split: Split):
        data = np.load(npz_path)

        key_x = f"{split}_images"
        key_y = f"{split}_labels"
        if key_x not in data or key_y not in data:
            raise KeyError(
                f"NPZ missing keys for split='{split}'. Need '{key_x}' and '{key_y}'. "
                f"Available keys: {list(data.keys())}"
            )

        self.images = data[key_x]  # uint8, (N,H,W) or (N,H,W,3)
        self.labels = data[key_y]  # (N,1) or (N,) or multi-label

        # infer channels
        if self.images.ndim == 3:
            self.n_channels = 1
        elif self.images.ndim == 4:
            self.n_channels = self.images.shape[-1]
        else:
            raise ValueError(f"Unexpected image shape: {self.images.shape}")

    # -----------------------------
    # MedMNIST from package
    # -----------------------------
    def _load_medmnist_pkg(self, name: str, split: Split, root: str, size: Optional[int], download: bool):
        try:
            import medmnist
            from medmnist import INFO
        except ImportError as e:
            raise ImportError("source='medmnist' requires: pip install medmnist") from e

        if name not in INFO:
            raise ValueError(f"Unknown MedMNIST dataset '{name}'. Valid keys include: {list(INFO.keys())[:10]} ...")

        info = INFO[name]
        DataClass = getattr(medmnist, info["python_class"])
        self.n_channels = info["n_channels"]

        self.med_ds = DataClass(split=split, root=root, download=download, size=size)

    # -----------------------------
    # Dataset interface
    # -----------------------------
    def __len__(self):
        if self.mode == "openi":
            return len(self.metadata)
        if self.med_source == "npz":
            return int(self.images.shape[0])
        return len(self.med_ds)

    def __getitem__(self, idx):
        if self.mode == "openi":
            return self._get_openi(idx)
        if self.med_source == "npz":
            return self._get_med_npz(idx)
        return self._get_med_pkg(idx)

    # -----------------------------
    # getters
    # -----------------------------
    def _get_openi(self, idx):
        row = self.metadata.iloc[idx]

        img = Image.open(row["image_path"]).convert("L")
        x = self.tf_gray(img)

        y = torch.tensor(int(row["label"]), dtype=torch.long)

        return x, y

    def _get_med_npz(self, idx):
        img = self.images[idx]
        label = self.labels[idx]

        # label to int (single-label)
        if hasattr(label, "shape"):
            label = int(np.array(label).reshape(-1)[0])
        else:
            label = int(label)

        # 3D volumes: (D,H,W) for 3D datasets only
        if self.is_3d and img.ndim == 3:
            vol = torch.from_numpy(img.astype(np.float32)) / 255.0
            vol = vol.unsqueeze(0)  # [1,D,H,W]
            if self.image_size_3d is not None:
                vol = F.interpolate(
                    vol.unsqueeze(0),
                    size=(self.image_size_3d, self.image_size_3d, self.image_size_3d),
                    mode="trilinear",
                    align_corners=False,
                ).squeeze(0)
            x = vol
        else:
            # image to tensor via PIL transforms (2D)
            if self.n_channels == 1:
                pil = Image.fromarray(img.squeeze(), mode="L")
                x = self.tf_gray(pil)
            else:
                pil = Image.fromarray(img, mode="RGB")
                x = self.tf_rgb(pil)

        y = torch.tensor(label, dtype=torch.long)
        return x, y

    def _get_med_pkg(self, idx):
        img, label = self.med_ds[idx]

        # label to int
        if hasattr(label, "shape"):
            label = int(np.array(label).reshape(-1)[0])
        else:
            label = int(label)

        # img numpy -> tensor (3D) or PIL (2D)
        if self.is_3d and isinstance(img, np.ndarray) and img.ndim == 3:
            vol = torch.from_numpy(img.astype(np.float32)) / 255.0
            vol = vol.unsqueeze(0)  # [1,D,H,W]
            if self.image_size_3d is not None:
                vol = F.interpolate(
                    vol.unsqueeze(0),
                    size=(self.image_size_3d, self.image_size_3d, self.image_size_3d),
                    mode="trilinear",
                    align_corners=False,
                ).squeeze(0)
            x = vol
        else:
            if self.n_channels == 1:
                pil = Image.fromarray(img.squeeze(), mode="L")
                x = self.tf_gray(pil)
            else:
                pil = Image.fromarray(img, mode="RGB")
                x = self.tf_rgb(pil)

        y = torch.tensor(label, dtype=torch.long)
        return x, y
