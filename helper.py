"""
helper 定义了工具函数, 用于辅助其他模块的功能实现

    @Time    : 2025/02/26
    @Author  : JackWang
    @File    : helper.py
    @IDE     : VsCode
"""

# Standard Library
import os
import sys
import random
import argparse
from io import StringIO
from pathlib import Path
from collections.abc import Callable

# Third-Party Library
import numpy as np
from loguru import logger
from rich.table import Table
from rich.console import Console
from torchmetrics import ConfusionMatrix
from scipy.optimize import linear_sum_assignment

hungarian_optimal_assignments: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]] = (
    linear_sum_assignment
)

# Torch Library
import torch

# My Library


def get_logger(log_file: Path, with_time: bool = True):
    global logger

    logger.remove()
    logger.add(
        log_file,
        level="DEBUG",
        format=f"{'{time:YYYY-D-MMMM@HH:mm:ss}' if with_time else ''}│ {{message}}",
    )
    logger.add(
        sys.stderr,
        level="DEBUG",
        format=f"{'{time:YYYY-D-MMMM@HH:mm:ss}' if with_time else ''}│ <level>{{message}}</level>",
    )

    return logger


def set_random_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def get_freer_gpu():
    os.system("nvidia-smi -q -d Memory |grep -A4 GPU | grep Used >tmp")
    memory_available = [int(x.split()[2]) for x in open("tmp", "r").readlines()]
    return np.argmin(memory_available)


def get_mapping_table_rich(mapping: torch.Tensor, cols_per_block: int = 10) -> str:
    """
    get_mapping_table_rich 将聚类簇与类标签之间的映射关系以表格的形式打印出来

    Args:
        mapping (torch.Tensor): 聚类簇与类标签之间的映射关系

    Returns:
        str: 字符串形式的表格
    """
    console = Console(file=StringIO(), force_terminal=False)
    original = range(len(mapping))
    mapped = mapping.cpu().numpy()

    for i in range(0, len(mapping), cols_per_block):
        table = Table(
            title=(
                f"Block {i//cols_per_block + 1}"
                if len(mapping) > cols_per_block
                else ""
            ),
            show_header=True,
        )
        table.add_column("No.", justify="center")

        for col_idx in range(i, min(i + cols_per_block, len(mapping))):
            table.add_column(str(col_idx), justify="center")

        table.add_row("Cluster ID", *[str(x) for x in original[i : i + cols_per_block]])
        table.add_row("Class Label", *[str(x) for x in mapped[i : i + cols_per_block]])

        console.print(table)

    return console.file.getvalue()


@torch.no_grad()
def calculate_unsupervised_clustering_accuracy(
    y_pred: torch.Tensor, y_true: torch.Tensor, num_classes: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    device = y_pred.device

    confusion_matrix: torch.Tensor = ConfusionMatrix(
        task="multiclass", num_classes=num_classes
    ).to(device)(y_pred, y_true)

    confusion_matrix = confusion_matrix.cpu().numpy()

    clsuter_id, class_label = hungarian_optimal_assignments(-confusion_matrix)

    optimal_mapping = np.zeros(num_classes, dtype=np.int64)
    optimal_mapping[clsuter_id] = class_label
    optimal_mapping = torch.tensor(optimal_mapping, dtype=torch.long).to(device)

    acc = confusion_matrix[clsuter_id, class_label].sum() / len(y_pred)

    return (
        optimal_mapping,
        optimal_mapping[y_pred],
        torch.from_numpy(confusion_matrix),
        acc,
    )


if __name__ == "__main__":
    print(get_freer_gpu())
