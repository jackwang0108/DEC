"""
DEC (Unsupervised Deep Embedding for Clustering Analysis) 主程序

    @Time    : 2025/02/25
    @Author  : JackWang
    @File    : main.py
    @IDE     : VsCode
"""

# Standard Library
import datetime
import argparse
from pathlib import Path

# Third-Party Library

# Torch Library
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# My Library
from network import DEC
from helper import get_args, get_logger, set_random_seed, get_freer_gpu
from datasets import (
    get_datasets,
    mnist_collate_fn,
    stl10_collate_fn,
    reuters_collate_fn,
)

device = torch.device(f"cuda:{get_freer_gpu()}" if torch.cuda.is_available() else "cpu")

log_dir = (
    Path(__file__).resolve().parent
    / f"logs/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
)
log_dir.mkdir(parents=True, exist_ok=True)
logger = get_logger(log_file=log_dir / "log.log")

logger.success("DEC: Unsupervised Deep Embedding for Clustering Analysis")


def sae_train_one_layer(
    dec: DEC, train_loader: DataLoader, num_epoch: int = 200
) -> DEC:

    auto_encoder = dec.auto_encoders[f"autoencoder {len(dec.auto_encoders)}"]

    loss_func = nn.MSELoss()
    optimizer = optim.SGD(auto_encoder.parameters(), lr=0.001, weight_decay=0)

    # freeze the weights of the previous encoders
    for layer in dec.non_linear_mapping[:-1]:
        for param in layer.parameters():
            param.requires_grad = False

    for epoch in range(num_epoch):

        tot_loss = 0

        loss: torch.FloatTensor
        image: torch.FloatTensor
        for image, _ in train_loader:

            image = image.to(device)

            # reconstruction the output of the previous layer
            with torch.no_grad():
                x = dec.non_linear_mapping[:-1](image)
            y = auto_encoder(x)
            loss = loss_func(y, x)

            tot_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        logger.info(
            f"\t\tEpoch {epoch + 1}/{num_epoch} Loss: {tot_loss / len(train_loader)}"
        )

    return dec


def parameter_initialization(dec: DEC, train_loader: DataLoader) -> DEC:
    """Step 1: Parameter Initialization"""
    logger.warning("Step 1: Parameter Initialization")

    loss_func = nn.MSELoss()

    # Sec.3.2: ... We initialize DEC with a stacked autoencoder (SAE) because ...
    logger.info("\tInitializing DEC with a stacked autoencoder (SAE)")

    dec.add_layer(i := next(iter(train_loader))[0].shape[1], o := 500, "first", device)
    logger.debug(f"\tTraining autoencoder [0]: {i} -> {o}")
    dec = sae_train_one_layer(dec, train_loader)

    return dec


def parameter_optimization(dec: DEC) -> DEC:
    """Step 2: Parameter Optimization"""
    pass


def testing(dec: DEC) -> DEC:
    pass


def main(args: argparse.Namespace):

    logger.info("Arguments:")
    for key, value in vars(args).items():
        logger.debug(f"\t{key}: {value}")

    set_random_seed(args.seed)

    train_set = get_datasets(args.dataset, split="train")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=(
            mnist_collate_fn
            if args.dataset == "MNIST"
            else (stl10_collate_fn if args.dataset == "STL-10" else reuters_collate_fn)
        ),
    )

    # Sec.3 Deep embedded clustering: ... . DEC has two phases: (1) parameter initialization with a deep autoencoder (Vincent et al., 2010) and (2) parameter optimization (i.e., clustering) ...

    dec = DEC()

    # Step 1: Parameter Initialization
    dec = parameter_initialization(dec, train_loader)


if __name__ == "__main__":
    main(get_args())
