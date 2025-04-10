import math
import torch
from .base_gate import BaseGate
import torch.nn.functional as F
from .base_gate import BaseGate
from .utils import limit_by_capacity
import torch.nn as nn
import torch.nn.functional as F

class AuxGate(BaseGate):

    def __init__(self,d_model, num_expert, world_size, top_k=2):
        super().__init__(num_expert, world_size)

        self.top_k = top_k
        self.n_routed_experts = world_size*num_expert
        self.routed_scaling_factor = 2.5
        self.n_group = 2
        self.topk_group = 1
        self.norm_topk_prob = False
        self.hidden_size=d_model

        self.gate = nn.Linear(d_model, self.tot_expert, bias = True)

        # self.weight = nn.Parameter(torch.empty((self.n_routed_experts,d_model)))
        self.register_buffer("e_score_correction_bias", torch.zeros((self.n_routed_experts)))

    @torch.no_grad()
    def get_topk_indices(self, scores):
        scores_for_choice = scores.view(-1, self.n_routed_experts) + self.e_score_correction_bias.unsqueeze(0)
        group_scores = (
            scores_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = scores_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        return topk_indices
    
    def forward(self, hidden_states):
        assert not torch.isnan(hidden_states).any(), "NaN in hidden_states"
        # assert not torch.isnan(self.weight).any(), "NaN in router weights"
        hidden_states = hidden_states.view(-1, self.hidden_size)
        router_logits=self.gate(hidden_states)
        # router_logits = F.linear(hidden_states.type(torch.float16), self.weight.type(torch.float16))
        scores = router_logits.sigmoid()
        topk_indices = self.get_topk_indices(scores)
        topk_weights = scores.gather(1, topk_indices)
        # dummy loss
        self.set_loss(torch.zeros(1, requires_grad=True).to(hidden_states.device))
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights