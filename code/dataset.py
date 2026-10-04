"""DeepWeeds data loading and split validation."""
from __future__ import annotations
import random
from pathlib import Path
import numpy as np
import pandas as pd

NUM_CLASSES = 9
CLASS_NAMES = ["Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia", "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def load_split(labels_dir: str | Path, fold: int = 0):
    if fold not in range(5): raise ValueError("fold must be in 0..4")
    root = Path(labels_dir)
    frames = tuple(pd.read_csv(root / f"{part}_subset{fold}.csv") for part in ("train", "val", "test"))
    for part, df in zip(("train", "val", "test"), frames):
        missing = {"Filename", "Label"} - set(df.columns)
        if missing: raise ValueError(f"{part} CSV missing columns: {sorted(missing)}")
        if df["Filename"].duplicated().any(): raise ValueError(f"duplicate Filename in {part}")
        labels = pd.to_numeric(df["Label"], errors="raise")
        if not labels.between(0, NUM_CLASSES - 1).all(): raise ValueError(f"invalid label in {part}")
    return frames

def check_split(train_df, val_df, test_df, images_dir: str | Path) -> dict:
    frames = {"train": train_df, "val": val_df, "test": test_df}
    names = {k: set(v["Filename"].astype(str)) for k, v in frames.items()}
    overlap = {"train_val": len(names["train"] & names["val"]), "train_test": len(names["train"] & names["test"]), "val_test": len(names["val"] & names["test"])}
    if any(overlap.values()): raise ValueError(f"split leakage detected: {overlap}")
    union = set().union(*names.values())
    if len(union) != 17509: raise ValueError(f"expected 17,509 unique images, found {len(union)}")
    root = Path(images_dir); missing = sorted(n for n in union if not (root / n).is_file())
    if missing: raise FileNotFoundError(f"{len(missing)} images missing; first: {missing[:5]}")
    result = {"n": {k: len(v) for k, v in frames.items()}, "per_class": {k: v["Label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0).astype(int).to_dict() for k, v in frames.items()}, "overlap": overlap, "union": len(union), "missing": 0}
    print(result); return result

def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    from torchvision import transforms as T
    norm = [T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
    if not train: return T.Compose([T.Resize(256), T.CenterCrop(img_size), *norm])
    prefix = [T.RandomResizedCrop(img_size), T.RandomHorizontalFlip()]
    if aug == "color": prefix.append(T.ColorJitter(.3, .3, .3, .1))
    elif aug == "trivial": prefix.append(T.TrivialAugmentWide())
    elif aug == "randaug": prefix.append(T.RandAugment())
    elif aug != "basic": raise ValueError(f"unknown augmentation: {aug}")
    return T.Compose([*prefix, *norm])

def _seed_worker(worker_id):
    import torch
    seed = torch.initial_seed() % 2**32; np.random.seed(seed); random.seed(seed)

class DeepWeedsDataset:
    def __init__(self, df, images_dir, transform=None):
        self.df = df.reset_index(drop=True).copy(); self.images_dir = Path(images_dir); self.transform = transform
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        from PIL import Image
        row = self.df.iloc[i]; name = str(row["Filename"])
        with Image.open(self.images_dir / name) as im:
            image = im.convert("RGB")
            if self.transform: image = self.transform(image)
        return image, int(row["Label"]), name

def make_loader(df, images_dir, transform, batch_size, train, sampler=None, num_workers=2):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    ds = DeepWeedsDataset(df, images_dir, transform); sample_obj = None
    if sampler not in (None, "balanced"): raise ValueError("sampler must be None or 'balanced'")
    if sampler == "balanced":
        counts = df["Label"].value_counts(); weights = df["Label"].map(lambda y: 1.0 / counts.loc[y]).to_numpy()
        sample_obj = WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double), len(weights), replacement=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=train and sample_obj is None, sampler=sample_obj, drop_last=train, pin_memory=torch.cuda.is_available(), num_workers=num_workers, worker_init_fn=_seed_worker, persistent_workers=num_workers > 0)
