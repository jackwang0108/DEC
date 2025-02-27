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
from pathlib import Path

# Third-Party Library
import numpy as np
from loguru import logger

# Torch Library
import torch

# My Library


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DEC (Unsupervised Deep Embedding for Clustering Analysis)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        choices=["MNIST", "STL-10", "REUTERS"],
        help="The dataset to use",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="The batch size for training",
    )
    parser.add_argument(
        "--num_epoch",
        type=int,
        default=200,
        help="The number of epochs to train",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=666,
        help="The random seed for reproducibility",
    )
    return parser.parse_args()


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


if __name__ == "__main__":
    print(get_freer_gpu())
