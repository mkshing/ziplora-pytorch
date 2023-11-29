from typing import Optional, Union
import torch
from torch import nn


class ZipLoRALinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        init_merger_value: Optional[float] = 1.0,
        init_merger_value_2: Optional[float] = 1.0,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.weight_1 = nn.Parameter(
            torch.zeros((out_features, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )
        self.weight_2 = nn.Parameter(
            torch.zeros((out_features, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )
        self.merger_1 = nn.Parameter(
            torch.ones((in_features,), device=device, dtype=dtype) * init_merger_value
        )
        self.merger_2 = nn.Parameter(
            torch.ones((in_features,), device=device, dtype=dtype) * init_merger_value_2
        )
        self.out_features = out_features
        self.in_features = in_features
        self.forward_type = "merge"

    def set_forward_type(self, type: str = "merge"):
        assert type in ["merge", "weight_1", "weight_2"]
        self.forward_type = type

    def compute_mergers_similarity(self):
        return (self.merger_1 * self.merger_2).abs().mean()

    def get_ziplora_weight(self):
        return self.merger_1 * self.weight_1 + self.merger_2 * self.weight_2

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.weight_1.dtype
        if self.forward_type == "merge":
            weight = self.get_ziplora_weight()
        elif self.forward_type == "weight_1":
            weight = self.weight_1
        elif self.forward_type == "weight_2":
            weight = self.weight_2
        else:
            raise ValueError(self.forward_type)
        hidden_states = nn.functional.linear(hidden_states.to(dtype), weight=weight)
        return hidden_states.to(orig_dtype)


class ZipLoRALinearLayerInference(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.zeros((out_features, in_features), device=device, dtype=dtype),
            requires_grad=False,
        )
        self.out_features = out_features
        self.in_features = in_features

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        orig_dtype = hidden_states.dtype
        dtype = self.weight.dtype
        hidden_states = nn.functional.linear(
            hidden_states.to(dtype), weight=self.weight
        )
        return hidden_states.to(orig_dtype)
