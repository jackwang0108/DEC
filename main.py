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
from colorama import Fore, Style

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

    loss_func = nn.MSELoss()
    optimizer = optim.SGD(
        dec.current_auto_encoder.parameters(), lr=0.001, weight_decay=0
    )

    for epoch in range(num_epoch):

        tot_loss = 0

        loss: torch.FloatTensor
        image: torch.FloatTensor
        for image, _ in train_loader:

            image = image.to(device)

            # reconstruction the output of the previous layer
            x = dec.final_encoder(image.to(device))
            y = dec.current_auto_encoder(x)
            loss = loss_func(y, x)

            tot_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 10 == 0:
            logger.info(
                f"\t\tEpoch [{epoch + 1:{len(str(num_epoch))}}/{num_epoch}], Loss: {Fore.CYAN}{tot_loss / len(train_loader):.4f}{Fore.RESET}"
            )

    return dec


def sae_final_finetune(dec: DEC, train_loader: DataLoader, num_epoch: int = 200) -> DEC:
    loss_func = nn.MSELoss()
    optimizer = optim.SGD(
        list(dec.final_encoder.parameters()) + list(dec.final_decoder.parameters()),
        lr=0.001,
        weight_decay=0,
    )

    for epoch in range(num_epoch):

        tot_loss = 0

        loss: torch.FloatTensor
        image: torch.FloatTensor
        for image, _ in train_loader:

            image = image.to(device)
            feature = dec.final_encoder(image)
            reconstructed_image = dec.final_decoder(feature)

            loss = loss_func(reconstructed_image, image)

            tot_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 10 == 0:
            logger.info(
                f"\t    Epoch [{epoch + 1:{len(str(num_epoch))}}/{num_epoch}], Loss: {Fore.CYAN}{tot_loss / len(train_loader):.4f}{Fore.RESET}"
            )

    return dec


def parameter_initialization(
    dec: DEC, train_loader: DataLoader, num_epoch: int = 1
) -> DEC:
    """Step 1: Parameter Initialization"""
    logger.warning("Step 1: Parameter Initialization")

    # Sec.3.2: ... We initialize DEC with a stacked autoencoder (SAE) because ...
    logger.info("\tInitializing DEC with a stacked autoencoder (SAE)")

    num_dims = [next(iter(train_loader))[0].shape[1], 500, 500, 2000, 10]

    # Sec.3.2: ... We initialize the SAE network layer by layer with each layer being a de-noising autoencoder trained to reconstruct the previous layer’s output after random corruption (Vincent et al., 2010) ...

    logger.info(f"\t{Fore.MAGENTA}[1] SAE layer-wise training{Fore.RESET}")
    for layer_idx, (in_dim, out_dim) in enumerate(zip(num_dims[:-1], num_dims[1:])):

        logger.debug(
            f"\t    Training autoencoder {layer_idx + 1}: {in_dim} -> {out_dim}"
        )

        pair_key = (
            "first"
            if layer_idx == 0
            else ("last" if layer_idx == len(num_dims) - 2 else "mid")
        )

        with dec.train_new_layer(
            in_dim=in_dim,
            hidden_dim=out_dim,
            freeze=True,
            which_pair=pair_key,
            device=device,
        ):

            dec = sae_train_one_layer(dec, train_loader, num_epoch)

    # Sec.3.2: ... After greedy layer-wise training, we concatenate all encoder layers followed by all decoder layers, in reverse layer-wise training order, to form a deep autoencoder and then finetune it to minimize reconstruction loss ...
    logger.info(f"\t{Fore.MAGENTA}[2] SAE deep autoencoder finetune{Fore.RESET}")

    for param in dec.final_encoder.parameters():
        param.requires_grad = True

    for param in dec.final_decoder.parameters():
        param.requires_grad = True

    dec = sae_final_finetune(dec, train_loader, num_epoch)

    return dec


def parameter_optimization(dec: DEC, num_epoch: int = 1) -> DEC:
    """Step 2: Parameter Optimization"""
    logger.warning("Step 2: Parameter Optimization")


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
    dec = parameter_initialization(dec, train_loader, num_epoch=args.num_epoch)

    torch.save(dec, log_dir / "dec.pth")

    # Step 2: Parameter Optimization
    dec = parameter_optimization(dec)

    # Step 3: Testing
    testing(dec)

    return dec


if __name__ == "__main__":
    main(get_args())
