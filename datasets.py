"""
datasets 定义了数据集的加载和预处理

    @Time    : 2025/02/25
    @Author  : JackWang
    @File    : datasets.py
    @IDE     : VsCode
"""

# Standard Library
import subprocess
from pathlib import Path
from typing import Literal, Optional
from collections.abc import Callable

# Third-Party Library
import numpy as np
import PIL.Image as PIMage
from skimage.feature import hog

# Torch Library
import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, STL10


# My Library

# TODO: 写完REUTERS数据集


class REUTERS(data.Dataset):
    def __init__(
        self,
        root: Path,
        download: Optional[bool] = True,
        transform: Optional[Callable] = None,
        split: Literal["train", "test"] = "train",
    ):
        if download and not (p := root / "REUTERS").exists():
            p.mkdir(parents=True, exist_ok=True)
            self._download_and_extract(root)
        elif not root.exists():
            raise FileNotFoundError(f"Dataset not found at {root}")

        self.split = split
        self.transform = transform
        raise NotImplementedError("REUTERS dataset not implemented yet")

    def __len__(self) -> int:
        raise NotImplementedError("REUTERS dataset not implemented yet")

    def __getitem__(self, idx: int) -> torch.Tensor:
        raise NotImplementedError("REUTERS dataset not implemented yet")

    def _download_and_extract(self, root: Path):
        url = "https://kdd.ics.uci.edu/databases/reuters21578/reuters21578.tar.gz"
        tar_path = root / "reuters21578.tar.gz"
        folder_path = root / "REUTERS"

        # Download the dataset
        subprocess.run(["wget", "-c", url, "-O", str(tar_path)], check=True)

        # Extract the dataset
        folder_path.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["tar", "-xf", str(tar_path), "-C", str(folder_path)], check=True
        )


def get_datasets(
    dataset: Literal["MNIST", "STL-10", "REUTERS"],
    split: Literal["train", "test"] = "train",
    root_dir: Optional[Path] = Path(__file__).resolve().parent / "data",
) -> MNIST | STL10 | REUTERS:
    assert dataset in ["MNIST", "STL-10", "REUTERS"], f"dataset {dataset} not supported"

    transform = transforms.Compose([transforms.ToTensor()])

    if dataset == "MNIST":
        return MNIST(
            root=root_dir, download=True, transform=transform, train=split == "train"
        )

    elif dataset == "STL-10":
        # Sec.4.1 Datasets-STL10: ... We also used the unlabeled set when training our auto-encoders ...
        return STL10(
            root=root_dir,
            download=True,
            transform=transform,
            split="train+unlabeled" if split == "train" else "test",
        )

    elif dataset == "REUTERS":
        return REUTERS(root=root_dir, download=True, transform=transform, split=split)


# Sec.4.1 Datasets-STL10: ... , we normalize all datasets so that \frac{1}{d}||x_i||_2^2 is approximately 1, ...
def normalize(image_representation: torch.FloatTensor) -> torch.FloatTensor:
    return (
        image_representation
        / torch.norm(image_representation, keepdim=True)
        * image_representation.nelement()
    )


def mnist_collate_fn(
    batch: list[tuple[torch.FloatTensor, int]]
) -> tuple[torch.FloatTensor, torch.LongTensor]:
    raw_images, raw_labels = zip(*batch)

    images = torch.stack([normalize(image.flatten()) for image in raw_images])
    labels = torch.tensor(raw_labels, dtype=torch.int64)

    return images, labels


def calculate_hog_feature(
    image: torch.FloatTensor,
    pixel_per_cell: tuple[int, int] = (8, 8),
    cell_per_block: tuple[int, int] = (1, 1),
    orientations: int = 6,
) -> torch.FloatTensor:
    gray_image = transforms.Grayscale()(image).squeeze().numpy()
    hog_feature = hog(
        gray_image,
        pixels_per_cell=pixel_per_cell,
        cells_per_block=cell_per_block,
        orientations=orientations,
        feature_vector=True,
    )
    return torch.tensor(hog_feature, dtype=image.dtype)


def calculate_color_histogram(
    image: torch.FloatTensor, bins=(8, 8, 8)
) -> torch.FloatTensor:
    hist: np.ndarray
    hist = np.histogramdd(
        image.permute(1, 2, 0).reshape(-1, 3),
        bins=bins,
        range=[(0, 256), (0, 256), (0, 256)],
    )[0]
    hist /= hist.sum()
    return torch.tensor(hist.flatten(), dtype=image.dtype)


def stl10_collate_fn(
    batch: list[tuple[torch.FloatTensor, int]]
) -> tuple[torch.FloatTensor, torch.LongTensor]:
    images, raw_labels = zip(*batch)

    images: torch.FloatTensor

    image_representations: list[torch.FloatTensor] = []

    for image in images:

        hog_feature = calculate_hog_feature(image)
        color_histogram = calculate_color_histogram(image)

        # Sec.4.1 Datasets-STL10: ... we concatenated HOG feature and a 8-by-8 color map to use as input to all algorithms ...
        image_representations.append(
            normalize(torch.cat([hog_feature, color_histogram]))
        )

    image_representations = torch.stack(image_representations)
    labels = torch.tensor(raw_labels, dtype=torch.int64)

    return image_representations, labels


def reuters_collate_fn(
    batch: list[tuple[PIMage.Image, int]]
) -> tuple[torch.FloatTensor, torch.LongTensor]:
    raise NotImplementedError("REUTERS dataset not implemented yet")


if __name__ == "__main__":

    label: torch.LongTensor
    image: torch.FloatTensor

    # test on mnist
    mnist = get_datasets("MNIST")
    dataloader = data.DataLoader(mnist, batch_size=4, collate_fn=mnist_collate_fn)
    for image, label in dataloader:
        print(image.shape, label.shape)
        print(image.norm(p=2, dim=1))
        break

    # test on stl10
    stl10 = get_datasets("STL-10")
    dataloader = data.DataLoader(stl10, batch_size=4, collate_fn=stl10_collate_fn)
    for image, label in dataloader:
        print(image.shape, label.shape)
        print(image.norm(p=2, dim=1))
        break

    # test on reuters
    # reuters = get_datasets("REUTERS")
    # print(type(reuters))
