"""FMF-Patch: Patch temporal encoder with factor-aware fusion mechanisms."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

SEQUENCE_LENGTH = 72
N_FEATURES = 49
FACTOR_GROUPS = {
    "demand": [48, 41, 44, 45, 46, 47],
    "operation": [4, 5, 12, 13, 42, 43],
    "fleet": list(range(34, 37)) + list(range(14, 34)),
    "context": [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 37, 38, 39, 40],
}


def static_features(capacity, unit_power):
    return torch.stack([capacity / 200.0, unit_power / 20.0], dim=1)


class PositionalEncoding(nn.Module):
    def __init__(self, dimension, max_length=160):
        super().__init__()
        encoding = torch.zeros(max_length, dimension)
        position = torch.arange(max_length).unsqueeze(1).float()
        divisor = torch.exp(torch.arange(0, dimension, 2).float() * (-math.log(10000.0) / dimension))
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.encoding[:, : x.size(1)]


class GatedResidualDelta(nn.Module):
    def __init__(self, dimension, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dimension, dimension), nn.GELU(), nn.Dropout(dropout), nn.Linear(dimension, dimension * 2))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        value, gate = self.net(x).chunk(2, dim=-1)
        return value * torch.sigmoid(gate)


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, dimension, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, dimension)
        self.fc2 = nn.Linear(dimension, dimension)
        self.gate = nn.Linear(dimension, dimension)
        self.projection = nn.Linear(dimension, dimension)
        self.skip = nn.Linear(input_dim, dimension) if input_dim != dimension else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dimension)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)
        nn.init.zeros_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

    def forward(self, x, context=None):
        hidden = F.elu(self.fc1(x))
        hidden = self.dropout(F.elu(self.fc2(hidden)))
        update = self.projection(torch.sigmoid(self.gate(hidden)) * hidden)
        return self.norm((context if context is not None else self.skip(x)) + update)


class FactorReadout(nn.Module):
    def __init__(self, dimension):
        super().__init__()
        self.time_pool = nn.Linear(dimension * 2, dimension)
        self.attention = nn.Linear(dimension, 1)
        self.group_mlps = nn.ModuleDict({name: nn.Sequential(nn.Linear(len(indices), dimension), nn.GELU(), nn.Linear(dimension, dimension)) for name, indices in FACTOR_GROUPS.items()})
        self.norms = nn.ModuleDict({name: nn.LayerNorm(dimension) for name in FACTOR_GROUPS})

    def forward(self, temporal, raw_features):
        weight = torch.softmax(self.attention(temporal).squeeze(-1), dim=-1)
        summary = self.time_pool(torch.cat([(temporal * weight.unsqueeze(-1)).sum(1), temporal[:, -1]], dim=-1))
        return {name: self.norms[name](summary + self.group_mlps[name](raw_features[:, -1, indices])) for name, indices in FACTOR_GROUPS.items()}


class PatchFront(nn.Module):
    def __init__(self, dimension=64, patch=12, stride=6, layers=2):
        super().__init__()
        self.patch, self.stride = patch, stride
        self.input_projection = nn.Linear(N_FEATURES, dimension)
        self.skip = nn.Linear(N_FEATURES, dimension)
        self.patch_embedding = nn.Linear(patch * dimension, dimension)
        self.position = PositionalEncoding(dimension)
        layer = nn.TransformerEncoderLayer(dimension, 4, 128, 0.1, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers)
        self.dropout = nn.Dropout(0.1)
        self.unpatch = nn.Linear(dimension, patch * dimension)
        self.output_norm = nn.LayerNorm(dimension)
        self.factor_readout = FactorReadout(dimension)
        self.static = nn.Sequential(nn.Linear(2, 16), nn.GELU())
        self.initial_state = nn.Sequential(nn.Linear(dimension * 2 + 16, dimension), nn.GELU(), nn.LayerNorm(dimension))
        self.pool, self.pool_attention = nn.Linear(dimension * 2, dimension), nn.Linear(dimension, 1)

    def forward(self, x, capacity, unit_power):
        batch, steps, _ = x.shape
        hidden = self.input_projection(x)
        patches = hidden.unfold(1, self.patch, self.stride).permute(0, 1, 3, 2).contiguous()
        tokens = self.encoder(self.dropout(self.position(self.patch_embedding(patches.view(batch, patches.size(1), -1)))))
        reconstructed = self.unpatch(tokens).view(batch, tokens.size(1), self.patch, -1)
        temporal, count = hidden.new_zeros(batch, steps, hidden.size(-1)), hidden.new_zeros(batch, steps, 1)
        for index in range(tokens.size(1)):
            start = index * self.stride
            temporal[:, start : start + self.patch] += reconstructed[:, index]
            count[:, start : start + self.patch] += 1
        temporal = self.output_norm(temporal / count.clamp_min(1.0) + self.skip(x))
        factors = self.factor_readout(temporal, x)
        static = self.static(static_features(capacity, unit_power))
        weight = torch.softmax(self.pool_attention(temporal).squeeze(-1), dim=-1)
        pooled = self.pool(torch.cat([temporal[:, -1], (temporal * weight.unsqueeze(-1)).sum(1)], dim=-1))
        state = self.initial_state(torch.cat([factors["demand"], factors["fleet"], static], dim=-1)) + pooled
        return temporal, factors, state, static


class FusionBlock(nn.Module):
    def __init__(self, dimension=64, use_multiscale=True, use_cross_factor=True, use_trend_residual=True):
        super().__init__()
        self.use_multiscale = use_multiscale
        self.use_cross_factor = use_cross_factor
        self.use_trend_residual = use_trend_residual
        if use_multiscale:
            self.fine, self.coarse = nn.Conv1d(dimension, dimension, 6, stride=3), nn.Conv1d(dimension, dimension, 18, stride=9)
            self.scale_mix, self.scale_condition, self.scale_gate = nn.Sequential(nn.Linear(dimension * 2, dimension), nn.GELU(), nn.Linear(dimension, dimension)), nn.Linear(dimension, dimension), GatedResidualDelta(dimension)
        if use_cross_factor:
            self.factor_attention = nn.MultiheadAttention(dimension, 1, dropout=0.1, batch_first=True)
            self.factor_norm1, self.factor_norm2 = nn.LayerNorm(dimension), nn.LayerNorm(dimension)
            self.factor_ff, self.factor_gate = nn.Sequential(nn.Linear(dimension, dimension * 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(dimension * 2, dimension)), GatedResidualDelta(dimension)
            nn.init.zeros_(self.factor_ff[-1].weight)
            nn.init.zeros_(self.factor_ff[-1].bias)
        if use_trend_residual:
            self.trend_readout, self.trend, self.residual, self.trend_gate = nn.Linear(dimension, 1), nn.Linear(SEQUENCE_LENGTH, dimension), nn.Linear(SEQUENCE_LENGTH, dimension), GatedResidualDelta(dimension)
        self.fusion, self.norm = GatedResidualNetwork(dimension * 3 + 16, dimension), nn.LayerNorm(dimension)

    def forward(self, state, temporal, factors, static):
        if self.use_multiscale:
            hidden = temporal.transpose(1, 2)
            scale = self.scale_gate(self.scale_mix(torch.cat([self.fine(hidden).mean(-1), self.coarse(hidden).mean(-1)], dim=-1)) + self.scale_condition(factors["demand"]))
            state = self.norm(state + scale)
        if self.use_cross_factor:
            tokens = torch.stack(list(factors.values()), dim=1)
            attended, _ = self.factor_attention(tokens, tokens, tokens, need_weights=False)
            factor_hidden = self.factor_norm1(tokens + attended)
            factor_hidden = self.factor_norm2(factor_hidden + self.factor_ff(factor_hidden))
            state = self.norm(state + self.factor_gate(factor_hidden.mean(1)))
        if self.use_trend_residual:
            series = self.trend_readout(temporal).squeeze(-1)
            trend = F.avg_pool1d(F.pad(series.unsqueeze(1), (12, 12), mode="replicate"), 25, 1).squeeze(1)
            state = self.norm(state + self.trend_gate(F.gelu(self.trend(trend) + self.residual(series - trend))))
        return self.fusion(torch.cat([state, factors["demand"], factors["fleet"], static], dim=-1), context=state)


class QuantileHead(nn.Module):
    def __init__(self, dimension, horizon):
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(nn.Linear(dimension, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, horizon * 5))

    def forward(self, hidden, capacity, last_demand):
        raw = self.net(hidden).view(hidden.size(0), self.horizon, 5)
        mu, alpha, q50 = F.softplus(raw[..., 0]) + 1e-3, F.softplus(raw[..., 1]) + 1e-3, F.softplus(raw[..., 2])
        q50, mu = 0.5 * q50 + 0.5 * last_demand.unsqueeze(1).clamp_min(0), 0.5 * mu + 0.5 * last_demand.unsqueeze(1).clamp_min(0)
        q80, q90 = q50 + F.softplus(raw[..., 3]), q50 + F.softplus(raw[..., 3]) + F.softplus(raw[..., 4])
        cap = capacity.unsqueeze(1)
        return {"mu": mu, "alpha": alpha, "q50": torch.minimum(q50, cap), "q80": torch.minimum(q80, cap), "q90": torch.minimum(q90, cap)}


class DirectFourHead(nn.Module):
    """Direct mu, alpha, q50, and q90 head used for speed comparisons."""

    def __init__(self, dimension, horizon):
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(nn.Linear(dimension, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, horizon * 4))

    def forward(self, hidden, capacity, last_demand):
        raw = self.net(hidden).view(hidden.size(0), self.horizon, 4)
        mu, alpha, q50 = F.softplus(raw[..., 0]) + 1e-3, F.softplus(raw[..., 1]) + 1e-3, F.softplus(raw[..., 2])
        q50 = 0.5 * q50 + 0.5 * last_demand.unsqueeze(1).clamp_min(0)
        mu = 0.5 * mu + 0.5 * last_demand.unsqueeze(1).clamp_min(0)
        q90 = q50 + F.softplus(raw[..., 3])
        cap = capacity.unsqueeze(1)
        return {"mu": mu, "alpha": alpha, "q50": torch.minimum(q50, cap), "q90": torch.minimum(q90, cap)}


class DirectQuantileHead(nn.Module):
    """Pure, ordered q50/q80/q90 forecasts with no distribution parameters."""

    def __init__(self, dimension, horizon, hidden_width=128, dropout=0.1, last_demand_anchor=0.5):
        super().__init__()
        self.horizon = horizon
        self.last_demand_anchor = last_demand_anchor
        self.net = nn.Sequential(nn.Linear(dimension, hidden_width), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_width, horizon * 3))

    def forward(self, hidden, capacity, last_demand):
        raw = self.net(hidden).view(hidden.size(0), self.horizon, 3)
        q50 = (1.0 - self.last_demand_anchor) * F.softplus(raw[..., 0]) + self.last_demand_anchor * last_demand.unsqueeze(1).clamp_min(0)
        q80 = q50 + F.softplus(raw[..., 1])
        q90 = q80 + F.softplus(raw[..., 2])
        cap = capacity.unsqueeze(1)
        return {"q50": torch.minimum(q50, cap), "q80": torch.minimum(q80, cap), "q90": torch.minimum(q90, cap)}


class ApproximateNBHead(nn.Module):
    """Two-parameter NB head with differentiable normal-approximate quantiles."""

    _Z90 = 1.2815515655446004

    def __init__(self, dimension, horizon):
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(nn.Linear(dimension, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, horizon * 2))

    def forward(self, hidden, capacity, last_demand):
        raw = self.net(hidden).view(hidden.size(0), self.horizon, 2)
        mu, alpha = F.softplus(raw[..., 0]) + 1e-3, F.softplus(raw[..., 1]) + 1e-3
        mu = 0.5 * mu + 0.5 * last_demand.unsqueeze(1).clamp_min(0)
        q50 = mu
        q90 = mu + self._Z90 * torch.sqrt(mu + mu.square() / alpha)
        cap = capacity.unsqueeze(1)
        return {"mu": mu, "alpha": alpha, "q50": torch.minimum(q50, cap), "q90": torch.minimum(q90, cap)}


class ApproximateNBThreeQuantileHead(nn.Module):
    """Two-parameter NB head whose 0.8 and 0.9 quantiles are derived from mu/alpha."""

    _Z80 = 0.8416212335729143
    _Z90 = 1.2815515655446004

    def __init__(self, dimension, horizon):
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(nn.Linear(dimension, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, horizon * 2))

    def forward(self, hidden, capacity, last_demand):
        raw = self.net(hidden).view(hidden.size(0), self.horizon, 2)
        mu, alpha = F.softplus(raw[..., 0]) + 1e-3, F.softplus(raw[..., 1]) + 1e-3
        mu = 0.5 * mu + 0.5 * last_demand.unsqueeze(1).clamp_min(0)
        standard_deviation = torch.sqrt(mu + mu.square() / alpha)
        q50 = mu
        q80 = mu + self._Z80 * standard_deviation
        q90 = mu + self._Z90 * standard_deviation
        cap = capacity.unsqueeze(1)
        return {"mu": mu, "alpha": alpha, "q50": torch.minimum(q50, cap), "q80": torch.minimum(q80, cap), "q90": torch.minimum(q90, cap)}


class PoissonHead(nn.Module):
    """Poisson head with differentiable normal-approximate training quantiles."""

    _Z80 = 0.8416212335729143
    _Z90 = 1.2815515655446004

    def __init__(self, dimension, horizon):
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(nn.Linear(dimension, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, horizon))

    def forward(self, hidden, capacity, last_demand):
        mu = F.softplus(self.net(hidden).view(hidden.size(0), self.horizon)) + 1e-3
        mu = 0.5 * mu + 0.5 * last_demand.unsqueeze(1).clamp_min(0)
        standard_deviation = torch.sqrt(mu)
        cap = capacity.unsqueeze(1)
        return {
            "mu": mu,
            "q50": torch.minimum(mu, cap),
            "q80": torch.minimum(mu + self._Z80 * standard_deviation, cap),
            "q90": torch.minimum(mu + self._Z90 * standard_deviation, cap),
        }


class ZINBHead(nn.Module):
    """Zero-inflated NB head with a differentiable quantile approximation."""

    _Z80 = 0.8416212335729143
    _Z90 = 1.2815515655446004

    def __init__(self, dimension, horizon):
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(nn.Linear(dimension, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, horizon * 3))

    def forward(self, hidden, capacity, last_demand):
        raw = self.net(hidden).view(hidden.size(0), self.horizon, 3)
        zero_prob = torch.sigmoid(raw[..., 0]).clamp(1e-5, 1.0 - 1e-5)
        mu, alpha = F.softplus(raw[..., 1]) + 1e-3, F.softplus(raw[..., 2]) + 1e-3
        mu = 0.5 * mu + 0.5 * last_demand.unsqueeze(1).clamp_min(0)
        standard_deviation = torch.sqrt(mu + mu.square() / alpha)
        nb_zero = torch.exp(torch.distributions.NegativeBinomial(total_count=alpha, logits=torch.log(alpha) - torch.log(mu)).log_prob(torch.zeros_like(mu)))
        zero_mass = zero_prob + (1.0 - zero_prob) * nb_zero

        def quantile(level):
            adjusted_level = ((level - zero_prob) / (1.0 - zero_prob)).clamp(1e-5, 1.0 - 1e-5)
            z_score = math.sqrt(2.0) * torch.erfinv(2.0 * adjusted_level - 1.0)
            value = mu + z_score * standard_deviation
            return torch.where(torch.full_like(zero_mass, level) <= zero_mass, torch.zeros_like(value), value)

        cap = capacity.unsqueeze(1)
        return {
            "mu": (1.0 - zero_prob) * mu,
            "nb_mu": mu,
            "alpha": alpha,
            "zero_prob": zero_prob,
            "q50": torch.minimum(quantile(0.5), cap),
            "q80": torch.minimum(quantile(0.8), cap),
            "q90": torch.minimum(quantile(0.9), cap),
        }


class TruncatedNBHead(ApproximateNBThreeQuantileHead):
    """NB parameters with training quantiles truncated at station capacity.

    The exact upper-truncated likelihood and PPF are evaluated by the benchmark,
    where the station-specific finite support is available.
    """


class Log1pGaussianHead(nn.Module):
    """Gaussian conditional law for log1p demand."""

    _Z80 = 0.8416212335729143
    _Z90 = 1.2815515655446004

    def __init__(self, dimension, horizon):
        super().__init__()
        self.horizon = horizon
        self.net = nn.Sequential(nn.Linear(dimension, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, horizon * 2))

    def forward(self, hidden, capacity, last_demand):
        raw = self.net(hidden).view(hidden.size(0), self.horizon, 2)
        log_mean, log_scale = raw[..., 0], F.softplus(raw[..., 1]) + 1e-3
        cap = capacity.unsqueeze(1)

        def quantile(z_score):
            return torch.expm1(log_mean + z_score * log_scale).clamp_min(0.0)

        return {
            "mu": torch.expm1(log_mean + 0.5 * log_scale.square()).clamp_min(0.0),
            "log_mean": log_mean,
            "log_scale": log_scale,
            "q50": torch.minimum(quantile(0.0), cap),
            "q80": torch.minimum(quantile(self._Z80), cap),
            "q90": torch.minimum(quantile(self._Z90), cap),
        }


class AdaptiveQuantileRouterHead(nn.Module):
    """Mix direct and NB-derived quantiles with a state-dependent shared gate."""

    _Z80 = 0.8416212335729143
    _Z90 = 1.2815515655446004

    def __init__(self, dimension, horizon):
        super().__init__()
        self.horizon = horizon
        self.distribution = nn.Sequential(nn.Linear(dimension, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, horizon * 2))
        self.quantiles = nn.Sequential(nn.Linear(dimension, 128), nn.GELU(), nn.Dropout(0.1), nn.Linear(128, horizon * 3))
        self.router = nn.Sequential(nn.Linear(dimension, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, hidden, capacity, last_demand):
        distribution_raw = self.distribution(hidden).view(hidden.size(0), self.horizon, 2)
        mu = F.softplus(distribution_raw[..., 0]) + 1e-3
        alpha = F.softplus(distribution_raw[..., 1]) + 1e-3
        mu = 0.5 * mu + 0.5 * last_demand.unsqueeze(1).clamp_min(0)
        standard_deviation = torch.sqrt(mu + mu.square() / alpha)
        nb_q50, nb_q80, nb_q90 = mu, mu + self._Z80 * standard_deviation, mu + self._Z90 * standard_deviation

        direct_raw = self.quantiles(hidden).view(hidden.size(0), self.horizon, 3)
        direct_q50 = 0.5 * F.softplus(direct_raw[..., 0]) + 0.5 * last_demand.unsqueeze(1).clamp_min(0)
        direct_q80 = direct_q50 + F.softplus(direct_raw[..., 1])
        direct_q90 = direct_q80 + F.softplus(direct_raw[..., 2])
        router_weight = torch.sigmoid(self.router(hidden))
        cap = capacity.unsqueeze(1)

        def mix(direct, distribution):
            return torch.minimum(router_weight * direct + (1.0 - router_weight) * distribution, cap)

        return {
            "mu": mu,
            "alpha": alpha,
            "q50": mix(direct_q50, nb_q50),
            "q80": mix(direct_q80, nb_q80),
            "q90": mix(direct_q90, nb_q90),
            "direct_q50": torch.minimum(direct_q50, cap),
            "direct_q80": torch.minimum(direct_q80, cap),
            "direct_q90": torch.minimum(direct_q90, cap),
            "nb_q50": torch.minimum(nb_q50, cap),
            "nb_q80": torch.minimum(nb_q80, cap),
            "nb_q90": torch.minimum(nb_q90, cap),
            "router_weight": router_weight,
        }


class FMFPatch(nn.Module):
    """Official FMF-Patch architecture for H=1 or H=3 forecasting."""

    def __init__(self, horizon=1, dimension=64, head_type="five", use_multiscale=True, use_cross_factor=True, use_trend_residual=True, head_width=128, head_dropout=0.1, last_demand_anchor=0.5):
        super().__init__()
        heads = {
            "five": QuantileHead,
            "direct_quantile": DirectQuantileHead,
            "four": DirectFourHead,
            "two_approx": ApproximateNBHead,
            "two_approx_q80": ApproximateNBThreeQuantileHead,
            "poisson": PoissonHead,
            "zinb": ZINBHead,
            "truncated_nb": TruncatedNBHead,
            "log1p_gaussian": Log1pGaussianHead,
            "adaptive_router": AdaptiveQuantileRouterHead,
        }
        if head_type not in heads:
            raise ValueError(f"Unknown head_type: {head_type}")
        self.front = PatchFront(dimension)
        self.block = FusionBlock(dimension, use_multiscale, use_cross_factor, use_trend_residual)
        if head_type == "direct_quantile":
            self.head = DirectQuantileHead(dimension + 2, horizon, head_width, head_dropout, last_demand_anchor)
        else:
            self.head = heads[head_type](dimension + 2, horizon)

    def forward(self, x, capacity, unit_power, last_demand, _module_count=None):
        temporal, factors, state, static = self.front(x[:, :, :N_FEATURES], capacity, unit_power)
        state = self.block(state, temporal, factors, static)
        return self.head(torch.cat([state, static_features(capacity, unit_power)], dim=-1), capacity, last_demand)
