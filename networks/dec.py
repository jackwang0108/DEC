"""
network 定义了DEC网络结构以及与DEC相关的函数

    @Time    : 2025/02/26
    @Author  : JackWang
    @File    : dec.py
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


class DEC(nn.Module):
    def __init__(self):
        super(DEC, self).__init__()

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
    """
    在分类任务中, 我们常说的是分类这个词 (classification), 即将一个样本划分到某一个类中

    但是对于聚类任务, 我们更关心的是样本和聚类中心 (即簇), 因此我们更常说的是分配 (assignment) 这个词, 即将一个样本分配到某一个簇中

    传统聚类（如K-means）中，每个样本会被**硬分配（hard assignment）**到唯一的簇, 即:
            z_i = argmin_k ||x_i − μ_k ||**2
    因为是argmin操作, 所以每个样本只能属于一个簇

    但是在DEC中, 我们使用的是**软分配（soft assignment）**. 所谓软分配的含义, 就是使用相似度/概率来描述样本属于的簇, 此时针对一个样本, 就可以得到一个相似度向量, 进行归一化操作之后, 使得和为1之后, 即可视为概率向量
    """
    distribution = ((feature.unsqueeze(dim=1) - centroids.unsqueeze(dim=0)) ** 2).sum(
        dim=2
    )

    qij = (1 + distribution / alpha) ** (-(alpha + 1) / 2)

    return qij / qij.sum(dim=1, keepdim=True)


@torch.no_grad()
def get_centroids(
    dec: DEC, train_loader: DataLoader, num_clusters: int
) -> torch.Tensor:

    device = next(dec.parameters()).device

    features = []
    image: torch.Tensor
    for _, image, _ in train_loader:
        image = image.to(device)
        features.append(dec.final_encoder(image))
    features = torch.cat(features, dim=0)

    _, cluster_centers = kmeans(
        X=features, num_clusters=num_clusters, distance="euclidean", device=device
    )

    return cluster_centers


def target_distribution(
    qij: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    DEC的基础是流型假设

    流型假设 (数学的讲)
        1. 数据观测维度 ≠ 真实维度：
            例如人脸图片（像素维度=3072 for 32x32x3）
            实际可能由**光照、姿态、表情等少量参数（~10维）**控制
        2. 流形假设（Manifold Hypothesis）：
            真实数据通常**集中**在嵌入高维空间的低维流形上

    换人话来说, 流型其实就是特征. 因此用计算机的话来解释流形学习就是:
        1. 数据观测维度 ≠ 真实维度
            数据观测维度信息密度太低了, 并且存在很多噪声
        2. 流形假设（Manifold Hypothesis）:
            特征在纯净的特征空间中是集中分布的

    对于DEC而言, 他就是用一个AutoEncoder来学习一个低维的流形 (特征)

    在此基础上, DEC的核心先验知识是: **初始的高置信度的聚类结果大部分都是对的**

    什么意思?

    我们现在通过SAE训练好的AutoEncoder提取到的特征已经足够好了, 使得KMeans聚类得到的结果已经蛮不错了, 我们在Optimization时候需要继续优化.

    例如, 对于一个给定的样本, 计算出来的qij中如果有一个值是0.7, 其他值加起来是0.3, 那么我们有充足的理由相信, 这个样本就是属于0.7这个簇的

    所以, 我们在优化阶段需要做的, 就是让qij的尖顶更加突出, 其余地方更加平缓
    """

    # qij 可以理解为概率分布, 代表了每个样本属于每个簇的概率
    # 对所有样本求和后得到的 fj 则描述了簇中的样本数量
    fj = qij.sum(dim=0, keepdim=True) + eps

    # 然后去除一下频率的影响, 本质上是考虑了类别不均衡, 但是很粗糙
    pij = (qij**2 + eps) / fj

    return (pij / (pij.sum(dim=1, keepdim=True) + eps)).detach()


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
        help="degrees of freedom of the Student’s t- distribution",
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

    dec = DEC()
    dec.train_new_layer(784, 500, which_pair="first")
    dec.train_new_layer(500, 500, which_pair="mid")
    dec.train_new_layer(500, 2000, which_pair="mid")
    dec.train_new_layer(2000, 10, which_pair="last")
    print(dec)

    image = torch.rand(4, 784)
    print(dec.final_encoder(image).shape)
    print(dec.auto_encoders["autoencoder 0"](image).shape)
