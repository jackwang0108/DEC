"""
IDEC (Improved Deep Embedded Clustering with Local Structure Preservation) 主程序

    @Time    : 2025/04/13
    @Author  : JackWang
    @File    : idec-main.py
    @IDE     : VsCode
"""

# Standard Library
import datetime
import argparse
from pathlib import Path

# Third-Party Library
from colorama import Fore


# Torch Library
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# My Library
from networks.idec import (
    IDEC,
    get_args,
    get_centroids,
    soft_assignment,
    target_distribution,
)
from src.helper import (
    get_logger,
    set_random_seed,
    get_freer_gpu,
    get_mapping_table_rich,
    calculate_unsupervised_clustering_accuracy,
)
from src.datasets import (
    get_datasets,
    mnist_collate_fn,
    stl10_collate_fn,
    reuters_collate_fn,
)


device = torch.device(f"cuda:{get_freer_gpu()}" if torch.cuda.is_available() else "cpu")

t = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_dir = Path(__file__).resolve().parent / f"logs/{t}"
log_dir.mkdir(parents=True, exist_ok=True)
weights_dir = log_dir.parent.parent / "weights"
weights_dir.mkdir(parents=True, exist_ok=True)
logger = get_logger(log_file=log_dir / "log.log")


def sae_train_one_layer(
    idec: IDEC, train_loader: DataLoader, num_epoch: int = 235
) -> IDEC:
    loss_func = nn.MSELoss()
    optimizer = optim.SGD(
        idec.current_auto_encoder.parameters(), lr=0.001, weight_decay=0
    )

    for epoch in range(num_epoch):

        epoch_loss = 0

        loss: torch.FloatTensor
        image: torch.FloatTensor
        for _, image, _ in train_loader:

            image = image.to(device)

            # reconstruction the output of the previous layer
            x = idec.final_encoder(image)
            y = idec.current_auto_encoder(x)
            loss = loss_func(y, x)

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 10 == 0:
            logger.info(
                f"\t\tEpoch [{epoch:{len(str(num_epoch))}}/{num_epoch}], Loss: {Fore.CYAN}{epoch_loss / len(train_loader):.4f}{Fore.RESET}"
            )
    return idec


def sae_final_finetune(
    idec: IDEC, train_loader: DataLoader, num_epoch: int = 470
) -> IDEC:

    logger.info(f"\t\tRemoving dropout layers from the entire autoencoder")
    idec = idec.remove_dropout()

    loss_func = nn.MSELoss()
    optimizer = optim.SGD(
        list(idec.final_encoder.parameters()) + list(idec.final_decoder.parameters()),
        lr=0.001,
    )

    logger.info(f"\t\tFinal finetuning the entire autoencoder")
    for epoch in range(num_epoch):

        epoch_loss = 0

        loss: torch.FloatTensor
        image: torch.FloatTensor
        for _, image, _ in train_loader:

            image = image.to(device)
            feature = idec.final_encoder(image)
            reconstructed_image = idec.final_decoder(feature)

            loss = loss_func(reconstructed_image, image)

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 10 == 0:
            logger.info(
                f"\t\t\tEpoch [{epoch:{len(str(epoch))}}/{num_epoch}], Loss: {Fore.CYAN}{epoch_loss / len(train_loader):.4f}{Fore.RESET}"
            )

    return idec


def parameter_initialization(
    idec: IDEC,
    train_loader: DataLoader,
    num_clusters: int,
    init_epochs: int = 235,
    finetune_epochs: int = 470,
    save_path: Path = None,
    pretrained_weights: Path = None,
) -> tuple[IDEC, torch.Tensor]:
    """Step 1: Parameter Initialization"""
    logger.warning("Step 1: Parameter Initialization")

    # Skip if pretrained_weights is provided:
    if pretrained_weights is None:
        trainer = sae_train_one_layer
        logger.info("\tInitializing IDEC with a stacked autoencoder (SAE)")
    else:
        trainer = lambda idec, train_loader, num_epoch: idec

        logger.info(f"\tLoading pretrained weights from {pretrained_weights}")

    num_dims = [next(iter(train_loader))[1].shape[1], 500, 500, 2000, 10]

    if pretrained_weights is None:
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

        with idec.train_new_layer(
            in_dim=in_dim,
            hidden_dim=out_dim,
            freeze=True,
            which_pair=pair_key,
            device=device,
        ):

            idec = trainer(idec, train_loader, init_epochs)

    if pretrained_weights is not None:
        saved_data = torch.load(pretrained_weights, map_location=device)
        idec = idec.remove_dropout()
        idec.load_state_dict(saved_data["model"])
        for param in idec.parameters():
            param.requires_grad = True
        return idec, saved_data["centroids"]

    logger.info(f"\t{Fore.MAGENTA}[2] SAE deep autoencoder finetune{Fore.RESET}")

    idec = sae_final_finetune(idec, train_loader, finetune_epochs)

    logger.info(
        f"\t{Fore.MAGENTA}[3] IDEC initializing initial centroids: k={num_clusters}{Fore.RESET}"
    )

    centroids = get_centroids(idec, train_loader, num_clusters)

    # save the pretrained weights if requested
    if save_path is not None:
        logger.info(
            f"\t{Fore.MAGENTA}Saving pretrained weights to {save_path}{Fore.RESET}"
        )
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": {
                    k: (v.detach().cpu().clone() if isinstance(v, torch.Tensor) else v)
                    for k, v in idec.state_dict().items()
                },
                "centroids": centroids.to("cpu"),
                "num_clusters": num_clusters,
                "dataset": train_loader.dataset.__class__.__name__,
                "training_at": t,
            },
            save_path,
        )

    return (idec, centroids)


def parameter_optimization(
    idec: IDEC,
    initial_centroids: torch.Tensor,
    train_loader: DataLoader,
    alpha: float = 1.0,
    gamma: float = 0.1,
    total: int = 0.001,
    num_epoch: int = 200,
) -> IDEC:
    # sourcery skip: low-code-quality
    """Step 2: Parameter Optimization"""
    logger.warning("Step 2: Parameter Optimization")

    # .to也是一个操作, 会导致centroids变为非叶子结点, 所以要先移动, 在转为Parameter
    # centroids = nn.Parameter(initial_centroids, requires_grad=True).to(device)
    centroids = nn.Parameter(
        initial_centroids.clone().detach().to(device), requires_grad=True
    )

    optimizer = optim.SGD(
        list(idec.final_encoder.parameters())
        + [centroids]
        + list(idec.final_decoder.parameters()),
        lr=0.001,
        weight_decay=0,
    )

    mse_loss = nn.MSELoss(reduction="mean")
    kl_divergence = nn.KLDivLoss(reduction="batchmean")

    logger.info(
        f"\tStopping Parameter Optimization when less than {Fore.CYAN}{total*100:.2f}%{Fore.RESET} of points change"
    )

    epoch = 0
    previous_preds = torch.ones(
        len(train_loader.dataset), dtype=torch.long, device=device
    )
    current_preds = torch.zeros(
        len(train_loader.dataset), dtype=torch.long, device=device
    )
    ground_truth = torch.zeros(
        len(train_loader.dataset), dtype=torch.long, device=device
    )

    changed_ratio = 1
    while changed_ratio > total or epoch < num_epoch:

        previous_preds = current_preds.clone()

        total_Lc = 0
        total_Lr = 0

        # Sec.3.3 ... Update target distribution. ..., Therefore, to avoid instability, P should not be updated at each iteration ...
        pij = torch.zeros(
            len(train_loader.dataset),
            idec.final_encoder(next(iter(train_loader))[1].to(device)).shape[1],
            device=device,
        )

        with torch.no_grad():
            for idx, image, label in train_loader:
                image = image.to(device)
                feature = idec.final_encoder(image)
                qij = soft_assignment(feature=feature, centroids=centroids, alpha=alpha)
                pij[idx] = target_distribution(qij)

        # Training
        loss: torch.Tensor
        image: torch.Tensor
        label: torch.Tensor
        feature: torch.Tensor
        for idx, image, label in train_loader:

            image = image.to(device)
            label = label.to(device)

            feature = idec.final_encoder(image)

            reconstructed_image = idec.final_decoder(feature)

            Lr = mse_loss(reconstructed_image, image)

            qij = soft_assignment(feature=feature, centroids=centroids, alpha=alpha)

            Lc = kl_divergence(torch.log(qij + 1e-10), pij[idx])

            loss = Lr + gamma * Lc

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_Lr += Lr.item()
            total_Lc += Lc.item()

            # update the current results
            current_preds[idx] = qij.argmax(dim=1)

            ground_truth[idx] = label

        changed_ratio = (current_preds != previous_preds).sum() / len(
            train_loader.dataset
        )

        optimal_mapping, relabelled_preds, confusion_matrix, acc = (
            calculate_unsupervised_clustering_accuracy(
                y_pred=current_preds,
                y_true=ground_truth,
                num_classes=centroids.shape[0],
            )
        )

        if epoch % 10 == 0:
            logger.info(
                f"\t\tEpoch [{epoch:{len(str(num_epoch))}}], "
                f"Lr: {Fore.CYAN}{total_Lr / len(train_loader):.5f}{Fore.RESET}, "
                f"Lc: {Fore.CYAN}{total_Lc / len(train_loader):.5f}{Fore.RESET}, "
                f"Loss: {Fore.CYAN}{(total_Lr + gamma * total_Lr) / len(train_loader):.5f}{Fore.RESET} (Lr + {gamma} * Lc), "
                f"Testing Accuracy: {Fore.CYAN}{acc * 100:.2f}%{Fore.RESET}"
                + (
                    ""
                    if epoch == 0
                    else f"{Fore.YELLOW} | {Fore.CYAN}{changed_ratio * 100:.2f}%{Fore.RESET} of points changed"
                )
                + f"{Fore.RESET}"
            )

            if epoch % 50 == 0:
                logger.info("\t\tCurrent Cluster-Class Mapping")
                for line in str(get_mapping_table_rich(optimal_mapping)).split("\n"):
                    logger.info(f"\t\t\t{line}")

        epoch += 1


def main(args: argparse.Namespace):

    logger.success(
        "IDEC: Improved Deep Embedded Clustering with Local Structure Preservation"
    )

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

    idec = IDEC()

    # Step 1: Parameter Initialization
    idec, centroids = parameter_initialization(
        idec,
        train_loader,
        num_clusters=args.num_clusters,
        init_epochs=args.init_epochs,
        finetune_epochs=args.finetune_epochs,
        save_path=(
            weights_dir / args.dataset / f"{t}.pth" if args.save_weights else None
        ),
        pretrained_weights=(
            Path(args.pretrain_weights) if args.pretrain_weights else None
        ),
    )

    # Step 2: Parameter Optimization
    idec = parameter_optimization(
        idec,
        centroids,
        train_loader,
        alpha=args.alpha,
        gamma=args.gamma,
        total=args.total,
        num_epoch=args.optimization_epochs,
    )

    return idec


if __name__ == "__main__":
    main(get_args())
