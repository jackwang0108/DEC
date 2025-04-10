"""
network 定义了DEC网络的结构

    @Time    : 2025/02/26
    @Author  : JackWang
    @File    : network.py
    @IDE     : VsCode
"""

# Standard Library
from typing import Literal
from contextlib import contextmanager

# Third-Party Library

# Torch Library
import torch
import torch.nn as nn

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
