"""
@author: ptrblck
"""

import torch
from torch import Tensor
import torch.nn as nn

from typing import Type, Any, Callable, Union, List, Optional

class PixelNorm(nn.Module):
    def __init__(
        self,
        feature_sizes:List[int],
        eps: float = 1e-5,
        momentum: float = 0.1,
        track_running_stats: bool = True,
        device=None,
        dtype=None
    ) -> None:        
        super().__init__()
        self.factory_kwargs = {'device': device, 'dtype': dtype}
        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats
        
        self.register_buffer('running_magnitude', torch.ones(feature_sizes[0], feature_sizes[1], feature_sizes[1], **self.factory_kwargs))
        self.running_magnitude: Optional[Tensor]
        self.register_buffer('num_batches_tracked',
                                torch.tensor(0, dtype=torch.long,
                                              **{k: v for k, v in self.factory_kwargs.items() if k != 'dtype'}))      

        self.reset_running_stats()

    def reset_running_stats(self) -> None:
        if self.track_running_stats:
            # if self.track_running_stats is on
            self.num_batches_tracked.zero_()  # type: ignore[union-attr,operator]

    def forward(self, input):
        
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        # calculate running estimates
        if self.training:
            magnitude =  torch.sqrt((input**2).mean([0]) + self.eps)
            n = input.numel() / input.size(1)
            with torch.no_grad():
                self.running_magnitude = exponential_average_factor * magnitude * n / (n - 1)\
                    + (1 - exponential_average_factor) * self.running_magnitude
        else:
            magnitude = self.running_magnitude
        input = (input / magnitude[None, :, :, :])
        return input

