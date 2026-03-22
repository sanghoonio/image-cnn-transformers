"""PASCAL VOC 2012 multi-label classification dataset."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]


class VOCClassification(Dataset):
    """PASCAL VOC 2012 multi-label classification dataset.

    Reads per-class label files from ImageSets/Main/{class}_{split}.txt.
    Each line: `image_id  {1, -1, 0}` (positive, negative, difficult).
    Difficult examples (0) are treated as negative.
    """

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform=None,
        indices: np.ndarray | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.transform = transform

        voc_root = self.root / "VOC2012"
        image_sets_dir = voc_root / "ImageSets" / "Main"
        self.image_dir = voc_root / "JPEGImages"

        # Read image IDs and labels from per-class files
        self.image_ids, self.labels = self._load_labels(image_sets_dir)

        # Apply subset indices if provided
        if indices is not None:
            self.image_ids = [self.image_ids[i] for i in indices]
            self.labels = self.labels[indices]

    def _load_labels(self, image_sets_dir: Path):
        """Parse per-class label files and build label matrix."""
        # Get image IDs from the first class file
        first_file = image_sets_dir / f"{VOC_CLASSES[0]}_{self.split}.txt"
        with open(first_file) as f:
            image_ids = [line.split()[0] for line in f.read().strip().splitlines()]

        n_images = len(image_ids)
        labels = np.zeros((n_images, len(VOC_CLASSES)), dtype=np.float32)

        for cls_idx, cls_name in enumerate(VOC_CLASSES):
            cls_file = image_sets_dir / f"{cls_name}_{self.split}.txt"
            with open(cls_file) as f:
                for img_idx, line in enumerate(f.read().strip().splitlines()):
                    parts = line.split()
                    flag = int(parts[1])
                    # 1 = positive, -1 = negative, 0 = difficult (treat as negative)
                    if flag == 1:
                        labels[img_idx, cls_idx] = 1.0

        return image_ids, labels

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = self.image_dir / f"{image_id}.jpg"
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.from_numpy(self.labels[idx])
        return image, label
