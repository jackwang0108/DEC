"""
network 定义了Improved DEC (IDEC) 网络结构以及与IDEC相关的函数

    @Time    : 2025/04/13
    @Author  : JackWang
    @File    : idec.py
    @IDE     : VsCode
"""

# Standard Library
import argparse
from typing import Literal
from contextlib import contextmanager

# Third-Party Library
from kmeans_pytorch import kmeans

# Torch Library
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# My Library


class IDEC(nn.Module):
    def __init__(self):
        super(IDEC, self).__init__()

        self.auto_encoders = nn.ModuleDict()
        self.final_encoder = nn.Sequential(nn.Identity())
        self.final_decoder = nn.Sequential(nn.Identity())

        self.current_auto_encoder = None

    @contextmanager
    def train_new_layer(
        self,
        in_dim: int,
        hidden_dim: int,
        freeze: bool = True,
        # Sec.3.2 Parameter initialization: ... except for g2 of the first pair (it needs to reconstruct input data that may have positive and negative values, such as zero-mean images) and g1 of the last pair (so the final data embedding retains full information ...
        which_pair: Literal["first", "last", "mid"] = True,
        device: torch.device = torch.device("cpu"),
    ):

        encoder = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU() if which_pair != "last" else nn.Identity(),
        ).to(device)

        decoder = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, in_dim),
            nn.ReLU() if which_pair != "first" else nn.Identity(),
        ).to(device)

        # Sec.4.3 Implementation: ... we initialize the weights to random numbers drawn from a zero-mean Gaussian distribution with a standard deviation of 0.01 ...
        nn.init.normal_(encoder[1].weight, mean=0, std=0.01)
        nn.init.normal_(decoder[1].weight, mean=0, std=0.01)

        self.auto_encoders.add_module(
            name=f"autoencoder {len(self.auto_encoders) + 1}",
            module=(cae := nn.Sequential(encoder, decoder)),
        )

        self.current_auto_encoder = cae

        yield self

        # freeze the weights of the trained encoder
        if freeze:
            for param in list(encoder.parameters()) + list(decoder.parameters()):
                param.requires_grad = False

        self.final_encoder.append(encoder)
        self.final_decoder.insert(0, decoder)

    def remove_dropout(self, module: nn.Module = None):
        """depth first traversal to remove dropout layers"""
        if module is None:
            module = self

        for name, child in module.named_children():
            if isinstance(child, nn.Dropout):
                setattr(module, name, nn.Identity())
            else:
                self.remove_dropout(child)

        if module == self:
            self.final_encoder.train()
            self.final_decoder.train()

            for param in list(self.final_encoder.parameters()) + list(
                self.final_decoder.parameters()
            ):
                param.requires_grad = True

        return module


def soft_assignment(
    feature: torch.Tensor, centroids: torch.Tensor, alpha: float = 1.0
) -> torch.Tensor:
    distribution = ((feature.unsqueeze(dim=1) - centroids.unsqueeze(dim=0)) ** 2).sum(
        dim=2
    )

    qij = (1 + distribution / alpha) ** (-(alpha + 1) / 2)

    return qij / qij.sum(dim=1, keepdim=True)


@torch.no_grad()
def get_centroids(
    idec: IDEC, train_loader: DataLoader, num_clusters: int
) -> torch.Tensor:

    device = next(idec.parameters()).device

    features = []
    image: torch.Tensor
    for _, image, _ in train_loader:
        image = image.to(device)
        features.append(idec.final_encoder(image))
    features = torch.cat(features, dim=0)

    _, cluster_centers = kmeans(
        X=features, num_clusters=num_clusters, distance="euclidean", device=device
    )

    return cluster_centers


def target_distribution(
    qij: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:

    # qij 可以理解为概率分布, 代表了每个样本属于每个簇的概率
    # 对所有样本求和后得到的 fj 则描述了簇中的样本数量
    fj = qij.sum(dim=0, keepdim=True) + eps

    # 然后去除一下频率的影响, 本质上是考虑了类别不均衡, 但是很粗糙
    pij = (qij**2 + eps) / fj

    return (pij / (pij.sum(dim=1, keepdim=True) + eps)).detach()


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IDEC (Improved Deep Embedded Clustering with Local Structure Preservation)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="MNIST",
        choices=["MNIST", "STL-10", "REUTERS"],
        help="The dataset to use",
    )
    parser.add_argument(
        "--pretrain_weights",
        type=str,
        default="",
        help="The path to the pre-trained weights of autoencoder",
    )
    parser.add_argument(
        "--save_weights",
        action="store_true",
        help="Whether to save the pre-trained weights of autoencoder",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="The batch size for training",
    )
    parser.add_argument(
        "--init_epochs",
        type=int,
        default=235,
        help="The number of epochs when training the autoencoder layer-wisely",
    )
    parser.add_argument(
        "--finetune_epochs",
        type=int,
        default=470,
        help="The number of epochs when finetuning the whole autoencoder",
    )
    parser.add_argument(
        "--optimization_epochs",
        type=int,
        default=200,
        help="The number of epochs for parameter optimization",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=666,
        help="The random seed for reproducibility",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="degrees of freedom of the Student’s t-distribution",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="The weight of KL divergence in the loss function",
    )
    parser.add_argument(
        "--num_clusters",
        type=int,
        default=10,
        help="The number of clusters to use",
    )
    parser.add_argument(
        "--total",
        type=float,
        default=0.001,
        help="The threshold of stopping optimization",
    )
    return parser.parse_args()


if __name__ == "__main__":

    idec = IDEC()
    idec.train_new_layer(784, 500, which_pair="first")
    idec.train_new_layer(500, 500, which_pair="mid")
    idec.train_new_layer(500, 2000, which_pair="mid")
    idec.train_new_layer(2000, 10, which_pair="last")
    print(idec)

    image = torch.rand(4, 784)
    print(idec.final_encoder(image).shape)
    print(idec.auto_encoders["autoencoder 0"](image).shape)
