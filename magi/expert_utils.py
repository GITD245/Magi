import torch
from magi.log import _print


class magi_expert:
    def __init__(self,magi_runtime):
        self.magi_expert_dic={}
        self.magi_runtime=magi_runtime
        self.rank=magi_runtime.get_rank()
        self.num_experts=magi_runtime.get_num_experts()

    # def _create_magi_expert_buffer(self,input_buffer,expert_size,global_expert_idx):
    #     layer=self.magi_profile.get_layer()
    #     new_expert_buffer=input_buffer.new_empty(expert_size);
    #     self.magi_expert_dic[(layer,global_expert_idx)]=new_expert_buffer
    #     return new_expert_buffer
    
    def _get_params(self,expert,out):
        offset = 0
        for n, p in expert.named_parameters():
            seg = out[offset:offset + p.numel()]
            offset += p.numel()
            seg.copy_(p.data.flatten())

    def get_magi_expert(self,out,global_expert_idx):
        expert=self.magi_expert_dic[(self.magi_runtime.get_layer(),global_expert_idx)]
        self._get_params(expert,out)
    
    def registe_magi_expert(self,new_expert_buffer,global_expert_idx,experts,get_params_flag=False):

        if get_params_flag:
            expert = experts[global_expert_idx%self.num_experts]
            self._get_params(expert,new_expert_buffer)
        
        self.magi_expert_dic[(self.magi_runtime.get_layer(),global_expert_idx)]=new_expert_buffer
        # self._print_rank_0(self.magi_expert_dic)