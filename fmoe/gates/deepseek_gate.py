import math
import torch
import torch.nn.functional as F
from .base_gate import BaseGate
from .utils import limit_by_capacity
import torch.nn as nn
import torch.nn.functional as F

class DeepSeekGate(BaseGate):
    def __init__(self, d_model, num_expert, world_size, top_k=6, gate_bias=True):
        super().__init__(num_expert, world_size)
        self.gate = nn.Linear(d_model, self.tot_expert, bias = gate_bias)
        self.top_k = top_k
        self.score_func="softmax"

    def forward(self,inp):
        scores = self.gate(inp)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores

        if bias is not None:
            scores = scores + self.bias

        if self.n_groups > 1:
            scores = scores.view(inp.size(0), self.n_groups, -1)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            mask = scores.new_ones(inp.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)

        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        return weights.type_as(inp), indices
