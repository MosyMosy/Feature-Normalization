import torch
import torch.nn as nn
from torch.nn import functional as F
import copy


class BatchNorm2d_plus(nn.BatchNorm2d):
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-5,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True):
        super(BatchNorm2d_plus, self).__init__(
            num_features, eps, momentum, affine, track_running_stats)
        self.output = None
        self.input = None
        self.before_affine = None

    def forward(self, input):
        self._check_input_dim(input)

        mean = input.mean([0, 2, 3])[None, :, None, None]
        var = input.var([0, 2, 3], unbiased=False)[None, :, None, None]
        self.input = input        
        self.before_affine = (self.input.clone() - mean)/ torch.sqrt(var + self.eps)        
        self.output = self.before_affine.clone() * self.weight[None, :, None, None] + self.bias[None, :, None, None]

        return self.output.clone()