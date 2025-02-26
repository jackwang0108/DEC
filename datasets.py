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

# Third-Party Library

# Torch Library
import torch
import torch.utils.data as data
from torchvision.datasets import MNIST, STL10


# My Library

# TODO: 写完REUTERS数据集


class REUTERS(data.Dataset):
    def __init__(
        self,
        root: Path,
        download: Optional[bool] = True,
        split: Literal["train", "test"] = "train",
    ):
        if download and not (p := root / "REUTERS").exists():
            p.mkdir(parents=True, exist_ok=True)
            self._download_and_extract(root)
        elif not root.exists():
            raise FileNotFoundError(f"Dataset not found at {root}")

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

    if dataset == "MNIST":
        return MNIST(root=root_dir, download=True, train=split == "train")

    elif dataset == "STL-10":
        # Sec.4.1 Datasets-STL10: We also used the unlabeled set when training our auto-encoders
        return STL10(
            root=root_dir,
            download=True,
            split="train+unlabeled" if split == "train" else "test",
        )

    elif dataset == "REUTERS":
        return REUTERS(root=root_dir, download=True, split=split)


if __name__ == "__main__":
    mnist = get_datasets("MNIST")
    print(next(iter(mnist)))

    stl10 = get_datasets("STL-10")
    print(next(iter(stl10)))

    reuters = get_datasets("REUTERS")
    print(type(reuters))
