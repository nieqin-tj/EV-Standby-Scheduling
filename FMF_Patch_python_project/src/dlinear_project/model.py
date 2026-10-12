"""DLinear decomposition/linear backbone and the archived two-quantile heads."""
import torch
from torch import nn
from torch.nn import functional as F


class MovingAverage(nn.Module):
    def __init__(self, kernel_size=25):
        super().__init__()
        self.kernel_size=kernel_size
        self.avg=nn.AvgPool1d(kernel_size=kernel_size,stride=1,padding=0)

    def forward(self,x):
        pad=(self.kernel_size-1)//2
        front=x[:,0:1,:].repeat(1,pad,1)
        end=x[:,-1:,:].repeat(1,pad,1)
        return self.avg(torch.cat([front,x,end],dim=1).permute(0,2,1)).permute(0,2,1)


class SeriesDecomposition(nn.Module):
    def __init__(self):
        super().__init__();self.moving_avg=MovingAverage(25)

    def forward(self,x):
        trend=self.moving_avg(x)
        return x-trend,trend


class DLinearBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.decompsition=SeriesDecomposition()
        self.Linear_Seasonal=nn.Linear(72,3)
        self.Linear_Trend=nn.Linear(72,3)
        self.Linear_Seasonal.weight=nn.Parameter(torch.ones(3,72)/72)
        self.Linear_Trend.weight=nn.Parameter(torch.ones(3,72)/72)

    def forward(self,x):
        seasonal,trend=self.decompsition(x)
        result=self.Linear_Seasonal(seasonal.permute(0,2,1))+self.Linear_Trend(trend.permute(0,2,1))
        return result.permute(0,2,1)


class DLinear(nn.Module):
    def __init__(self,mean,std):
        super().__init__()
        self.backbone=DLinearBackbone()
        self.q50_head=nn.Linear(49,1)
        self.delta90_head=nn.Linear(49,1)
        with torch.no_grad():
            self.q50_head.weight.zero_();self.q50_head.weight[0,-1]=1.;self.q50_head.bias.zero_()
        self.target_mean=float(mean[-1]);self.target_std=float(std[-1])

    def forward(self,x,capacity,unit_power=None,last_demand=None):
        features=self.backbone(x)
        median=self.q50_head(features).squeeze(-1)
        upper=median+F.softplus(self.delta90_head(features).squeeze(-1))
        return {name:torch.minimum((value*self.target_std+self.target_mean).clamp_min(0),capacity[:,None])
                for name,value in [('q50',median),('q90',upper)]}
