"""
DEC (Unsupervised Deep Embedding for Clustering Analysis) 主程序

读文章和写代码是两件事

写代码就很简单了, 只需要把文章的流程搞明白
读文章, 读到什么程度才算把文章彻底读透了: 背后的先验知识全部看懂了

DEC一共就两步:
    1. Parameter Initialization
    2. Parameter Optimization

代码逻辑:
    1. 数据准备 (数据集)
    2. 逐层训练SAE
        2.1 单独训练每一层
        2.2 联合训练所有层
    3. 初始化聚类中心
        3.1 训练集里所有图片过一遍训练好的Encoder得到特征
        3.2 用kmeans聚类
    4. 参数优化
        4.1 每个样本提取特征
        4.2 特征和每个类中心计算qij
        4.3 根据qij计算出来pij
        4.4 利用pij和qij计算KL散度
        4.5 需要统计相邻的两个epoch之间有多少个样本改变了

    @Time    : 2025/02/25
    @Author  : JackWang
    @File    : dec-main.py
    @IDE     : VsCode
"""

# Standard Library
import datetime
import argparse
from pathlib import Path

# Third-Party Library
import numpy as np
from colorama import Fore


# Torch Library
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# My Library
from networks.dec import (
    DEC,
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
    dec: DEC, train_loader: DataLoader, num_epoch: int = 235
) -> DEC:
    loss_func = nn.MSELoss()
    optimizer = optim.SGD(
        dec.current_auto_encoder.parameters(), lr=0.001, weight_decay=0
    )

    for epoch in range(num_epoch):

        epoch_loss = 0

        loss: torch.FloatTensor
        image: torch.FloatTensor
        for _, image, _ in train_loader:

            image = image.to(device)

            # reconstruction the output of the previous layer
            x = dec.final_encoder(image)
            y = dec.current_auto_encoder(x)
            loss = loss_func(y, x)

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 10 == 0:
            logger.info(
                f"\t\tEpoch [{epoch:{len(str(num_epoch))}}/{num_epoch}], Loss: {Fore.CYAN}{epoch_loss / len(train_loader):.4f}{Fore.RESET}"
            )
    return dec


def sae_final_finetune(dec: DEC, train_loader: DataLoader, num_epoch: int = 470) -> DEC:

    # Sec.4.3 ... The entire deep autoencoder is further finetuned for 100000 iterations without dropout ...
    logger.info(f"\t\tRemoving dropout layers from the entire autoencoder")
    dec = dec.remove_dropout()

    loss_func = nn.MSELoss()
    optimizer = optim.SGD(
        list(dec.final_encoder.parameters()) + list(dec.final_decoder.parameters()),
        lr=0.001,
    )

    logger.info(f"\t\tFinal finetuning the entire autoencoder")
    for epoch in range(num_epoch):

        epoch_loss = 0

        loss: torch.FloatTensor
        image: torch.FloatTensor
        for _, image, _ in train_loader:

            image = image.to(device)
            feature = dec.final_encoder(image)
            reconstructed_image = dec.final_decoder(feature)

            loss = loss_func(reconstructed_image, image)

            epoch_loss += loss.item()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if epoch % 10 == 0:
            logger.info(
                f"\t\t\tEpoch [{epoch:{len(str(epoch))}}/{num_epoch}], Loss: {Fore.CYAN}{epoch_loss / len(train_loader):.4f}{Fore.RESET}"
            )

    return dec


def parameter_initialization(
    dec: DEC,
    train_loader: DataLoader,
    num_clusters: int,
    init_epochs: int = 235,
    finetune_epochs: int = 470,
    save_path: Path = None,
    pretrained_weights: Path = None,
) -> tuple[DEC, torch.Tensor]:
    """Step 1: Parameter Initialization"""
    logger.warning("Step 1: Parameter Initialization")

    # Skip if pretrained_weights is provided:
    if pretrained_weights is None:
        trainer = sae_train_one_layer

        # Sec.3.2: ... We initialize DEC with a stacked autoencoder (SAE) because ...
        logger.info("\tInitializing DEC with a stacked autoencoder (SAE)")
    else:
        trainer = lambda dec, train_loader, num_epoch: dec

        logger.info(f"\tLoading pretrained weights from {pretrained_weights}")

    num_dims = [next(iter(train_loader))[1].shape[1], 500, 500, 2000, 10]

    # Sec.3.2: ... We initialize the SAE network layer by layer with each layer being a de-noising autoencoder trained to reconstruct the previous layer’s output after random corruption (Vincent et al., 2010) ...
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

        with dec.train_new_layer(
            in_dim=in_dim,
            hidden_dim=out_dim,
            freeze=True,
            which_pair=pair_key,
            device=device,
        ):

            dec = trainer(dec, train_loader, init_epochs)

    if pretrained_weights is not None:
        saved_data = torch.load(pretrained_weights, map_location=device)
        dec = dec.remove_dropout()
        dec.load_state_dict(saved_data["model"])
        for param in dec.parameters():
            param.requires_grad = True
        return dec, saved_data["centroids"]

    # Sec.3.2: ... After greedy layer-wise training, we concatenate all encoder layers followed by all decoder layers, in reverse layer-wise training order, to form a deep autoencoder and then finetune it to minimize reconstruction loss ...
    logger.info(f"\t{Fore.MAGENTA}[2] SAE deep autoencoder finetune{Fore.RESET}")

    dec = sae_final_finetune(dec, train_loader, finetune_epochs)

    # Sec.3.2: ... To initialize the cluster centers, we pass the data through the initialized DNN to get embedded data points and then perform standard k-means clustering ...
    logger.info(
        f"\t{Fore.MAGENTA}[3] DEC initializing initial centroids: k={num_clusters}{Fore.RESET}"
    )

    centroids = get_centroids(dec, train_loader, num_clusters)

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
                    for k, v in dec.state_dict().items()
                },
                "centroids": centroids.to("cpu"),
                "num_clusters": num_clusters,
                "dataset": train_loader.dataset.__class__.__name__,
                "training_at": t,
            },
            save_path,
        )

    return (dec, centroids)


def parameter_optimization(
    dec: DEC,
    initial_centroids: torch.Tensor,
    train_loader: DataLoader,
    alpha: float = 1.0,
    total: int = 0.001,
    num_epoch: int = 200,
) -> DEC:
    # sourcery skip: low-code-quality
    """Step 2: Parameter Optimization"""
    logger.warning("Step 2: Parameter Optimization")

    # .to也是一个操作, 会导致centroids变为非叶子结点, 所以要先移动, 在转为Parameter
    # centroids = nn.Parameter(initial_centroids, requires_grad=True).to(device)
    centroids = nn.Parameter(
        initial_centroids.clone().detach().to(device), requires_grad=True
    )

    optimizer = optim.SGD(
        list(dec.final_encoder.parameters()) + [centroids],
        lr=0.001,
        weight_decay=0,
    )

    kl_divergence = nn.KLDivLoss(reduction="batchmean")

    # Sec.3.1.3: ... For the purpose of discovering cluster assignments, we stop our procedure when less than tol% of points change cluster assignment between two consecutive iterations. ...
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

        total_loss = 0

        # Training
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
            current_preds[idx] = qij.argmax(dim=1)
            # current_results[idx] = get_preds(qij=qij, centroids=centroids)

            ground_truth[idx] = label

        changed_ratio = (current_preds != previous_preds).sum() / len(
            train_loader.dataset
        )

        # Sec.4.2: ... For all algorithms we set the number of clusters to the number of ground-truth categories and evaluate performance with unsupervised clustering accuracy (ACC) ...
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
                f"Loss: {Fore.CYAN}{total_loss / len(train_loader):.5f}{Fore.RESET}, "
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

    logger.success("DEC: Unsupervised Deep Embedding for Clustering Analysis")

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
    dec, centroids = parameter_initialization(
        dec,
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
    dec = parameter_optimization(
        dec,
        centroids,
        train_loader,
        alpha=args.alpha,
        total=args.total,
        num_epoch=args.finetune_epochs,
    )

    return dec


if __name__ == "__main__":
    main(get_args())
