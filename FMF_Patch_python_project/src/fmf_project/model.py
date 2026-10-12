"""Exact archived full FMF-Patch architecture and its two-quantile output head."""
import torch
from torch import nn
from torch.nn import functional as F
from .vendor.fmf_model import FMFPatch


class DualQuantileHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(98, 256), nn.GELU(), nn.Dropout(.05), nn.Linear(256, 6))

    def forward(self, hidden, capacity, last_demand):
        raw = self.net(hidden).reshape(len(hidden), 3, 2)
        q50 = .75 * F.softplus(raw[..., 0]) + .25 * last_demand[:, None].clamp_min(0)
        q90 = q50 + F.softplus(raw[..., 1])
        return dict(q50=torch.minimum(q50, capacity[:, None]), q90=torch.minimum(q90, capacity[:, None]))


def build_model():
    network = FMFPatch(horizon=3, dimension=96, head_type="direct_quantile",
                       head_width=256, head_dropout=.05, last_demand_anchor=.25)
    network.head = DualQuantileHead()
    return network
