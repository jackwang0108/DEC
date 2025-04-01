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
from kmeans_pytorch import kmeans

# Torch Library
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# My Library
from network import DEC
from helper import (
    get_args,
    get_logger,
    set_random_seed,
    get_freer_gpu,
    calculate_accuracy,
)
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
        for _, image, _ in train_loader:

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
        for _, image, _ in train_loader:

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


@torch.no_grad()
def dec_get_centroids(
    dec: DEC, train_loader: DataLoader, num_clusters: int
) -> torch.Tensor:

    features = []
    for _, image, _ in train_loader:
        image = image.to(device)
        features.append(dec.final_encoder(image))
    features = torch.cat(features, dim=0)

    _, cluster_centers = kmeans(
        X=features,
        num_clusters=num_clusters,
        distance="euclidean",
        device=torch.device("cuda:0"),
    )

    return cluster_centers


def parameter_initialization(
    dec: DEC, train_loader: DataLoader, num_clusters: int, num_epoch: int = 1
) -> tuple[DEC, torch.Tensor]:
    """Step 1: Parameter Initialization"""
    logger.warning("Step 1: Parameter Initialization")

    # Sec.3.2: ... We initialize DEC with a stacked autoencoder (SAE) because ...
    logger.info("\tInitializing DEC with a stacked autoencoder (SAE)")

    num_dims = [next(iter(train_loader))[1].shape[1], 500, 500, 2000, 10]

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

    # Sec.3.2: ... To initialize the cluster centers, we pass the data through the initialized DNN to get embedded data points and then perform standard k-means clustering ...
    logger.info(
        f"\t{Fore.MAGENTA}[3] DEC initializing initial centroids: k={num_clusters}{Fore.RESET}"
    )

    centroids = dec_get_centroids(dec, train_loader, num_clusters)

    return (dec, centroids)


def soft_assignment(
    feature: torch.Tensor, centroids: torch.Tensor, alpha: float = 1.0
) -> torch.Tensor:
    squared_distance = (feature.unsqueeze(dim=1) - centroids.unsqueeze(dim=0)).norm(
        p=2, dim=1
    )

    upper_parts = (1 + squared_distance / alpha) ** (-(1 + alpha / 2))

    qij = upper_parts / upper_parts.sum(dim=1, keepdim=True)

    return qij


def target_distribution(qij: torch.Tensor) -> torch.Tensor:
    fj = qij.sum(dim=1, keepdim=True)

    upper_parts = qij**2 / fj

    pij = upper_parts / upper_parts.sum(dim=1, keepdim=True)

    return pij


@torch.no_grad()
def test_epoch(
    dec: DEC, centroids: torch.Tensor, alpha: float, test_loader: DataLoader
) -> float:
    """Test the model on the test set"""
    dec.eval()

    accuracies = []
    image: torch.Tensor
    label: torch.Tensor
    feature: torch.Tensor
    for _, image, label in test_loader:
        feature = dec.final_encoder(image.to(device))

        accuracies.append(
            calculate_accuracy(
                y_true=label.to(device),
                y_pred=soft_assignment(
                    feature=feature, centroids=centroids, alpha=alpha
                ).argmax(dim=1),
            )
        )

    dec.train()
    return sum(accuracies) / len(accuracies)


def parameter_optimization(
    dec: DEC,
    initial_centroids: torch.Tensor,
    train_loader: DataLoader,
    test_loader: DataLoader,
    alpha: float = 1.0,
    total: int = 1,
) -> DEC:
    """Step 2: Parameter Optimization"""
    logger.warning("Step 2: Parameter Optimization")

    optimizer = optim.SGD(
        dec.final_encoder.parameters(), lr=0.001, weight_decay=0, momentum=0
    )

    kl_divergence = nn.KLDivLoss(reduction="batchmean")

    centroids = nn.Parameter(initial_centroids, requires_grad=True).to(device)

    # Sec.3.1.3: ... For the purpose of discovering cluster assignments, we stop our procedure when less than tol% of points change cluster assignment between two consecutive iterations. ...
    logger.info(
        f"\tStopping Parameter Optimization when less than {Fore.CYAN}{total*100:.2f}%{Fore.RESET} of points change"
    )

    epoch = 0
    previous_results = torch.ones(
        len(train_loader.dataset), dtype=torch.long, device=device
    )
    current_results = torch.zeros(
        len(train_loader.dataset), dtype=torch.long, device=device
    )

    changed_ratio = 1
    while changed_ratio > total:

        previous_results = current_results.clone()

        total_loss = 0

        accuracies = []

        loss: torch.Tensor
        image: torch.Tensor
        label: torch.Tensor
        feature: torch.Tensor
        for idx, image, label in train_loader:

            image = image.to(device)
            label = label.to(device)

            feature = dec.final_encoder(image)

            # Sec.3.1.1: ... Following van der Maaten & Hinton (2008) we use the Student’s t-distribution as a kernel to measure the similarity between embedded point zi and centroid µj:
            qij = soft_assignment(feature=feature, centroids=centroids, alpha=alpha)

            # Sec.3.1.2: ... In our experiments, we compute pi by first raising qi to the second power and then normalizing by frequency per cluster: ...
            pij = target_distribution(qij)

            # https://www.zhihu.com/question/384982085/answer/3111848737
            loss = kl_divergence(torch.log(qij + 1e-10), pij)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # update the current results
            current_results[idx] = qij.argmax(dim=1)

            accuracies.append(
                calculate_accuracy(
                    y_true=label,
                    y_pred=current_results[idx],
                )
            )

        changed_ratio = (current_results != previous_results).sum() / len(
            train_loader.dataset
        )
        logger.info(
            f"\t\tEpoch [{epoch + 1}], "
            f"Loss: {Fore.CYAN}{total_loss / len(train_loader):.5f}{Fore.RESET}, "
            f"Training Accuracy: {Fore.CYAN}{sum(accuracies) / len(accuracies):.2f}%{Fore.RESET}, "
            f"Testing Accuracy: {Fore.CYAN}{test_epoch(dec, centroids, alpha, test_loader):.2f}%{Fore.RESET}"
            + (
                ""
                if epoch == 0
                else f"{Fore.YELLOW} | {Fore.CYAN}{changed_ratio:.2f}%{Fore.RESET} of points changed"
            )
            + f"{Fore.RESET}"
        )

        epoch += 1


def testing(
    dec: DEC,
) -> DEC:
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

    test_set = get_datasets(args.dataset, split="test")
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=(
            mnist_collate_fn
            if args.dataset == "MNIST"
            else (stl10_collate_fn if args.dataset == "STL-10" else reuters_collate_fn)
        ),
    )

    # Sec.3 Deep embedded clustering: ... . DEC has two phases: (1) parameter initialization with a deep autoencoder (Vincent et al., 2010) and (2) parameter optimization (i.e., clustering) ...

    dec = DEC()

    # Step 1: Parameter Initialization
    dec, centroids = parameter_initialization(
        dec, train_loader, num_clusters=args.num_clusters, num_epoch=args.num_epoch
    )

    torch.save(dec, log_dir / "dec.pth")

    # Step 2: Parameter Optimization
    dec = parameter_optimization(
        dec,
        centroids,
        train_loader,
        test_loader,
        alpha=args.alpha,
        total=args.total,
    )

    # Step 3: Testing
    testing(dec)

    return dec


if __name__ == "__main__":
    main(get_args())
