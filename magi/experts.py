import torch
import copy
from magi import log
from magi import expert_utils

class magi_expert:
    def __init__(self,magi_runtime):
        self.magi_runtime=magi_runtime
        self.rank=magi_runtime.rank
        self.num_experts=magi_runtime.num_experts
        self.world_size=magi_runtime.world_size
        self.d_model=magi_runtime.d_model

        self.magi_expert_dic=dict()
        self.pl_experts=list()
    # not used
    def set_add_expert_fn(self,add_expert_fn):
        self.add_expert_fn=add_expert_fn
    
    def new_buffer(self,input_buf)->torch.Tensor:
        return input_buf.new_empty({self.expert_size});

    def set_expert_size(self,expert_size):
        self.expert_size=expert_size
    # not used

    def set_experts(self,experts):
        self.pl_experts.append(experts)

    def registe_magi_expert(self,new_expert_buffer,global_expert_idx,send_expert_flag):
        layer=self.magi_runtime.layer
        expert=self.pl_experts[self.magi_runtime.layer][global_expert_idx%self.num_experts]
        if send_expert_flag:
            # send part
            expert_utils.get_params(expert,new_expert_buffer)
        else:
            # receive part
            new_expert=copy.deepcopy(self.pl_experts[layer][-1])
            for p in new_expert.parameters():
                setattr(p, "dp_comm", 'none')
            self.pl_experts[layer].append(new_expert)
            self.magi_expert_dic[(layer,global_expert_idx)]=len(self.pl_experts[layer])-1

    def get_magi_expert_idx(self,global_expert_idx)->int:
        return self.magi_expert_dic[(self.magi_runtime.layer,global_expert_idx)]
         
    def push_magi_expert(self,buffer,global_expert_idx):
        stored_idx=self.magi_expert_dic[(self.magi_runtime.layer,global_expert_idx)]
        expert_utils.push_params(buffer,self.pl_experts[self.magi_runtime.layer][stored_idx])

    # def get_magi_expert_params(self,experts,global_expert_idx,input_buf,layer=-1)->torch.Tensor:
    #     if layer==-1:
    #         layer=self.magi_runtime.layer
    #     t = self.new_buffer(input_buf)

    #     expert_utils.get_params(experts[self.get_magi_expert_idx(global_expert_idx)],t)
    #     return t

    def del_magi_expert(self,layer,global_expert_idx):   
        self.pl_experts[layer].pop(self.magi_expert_dic[(layer,global_expert_idx)])
        # update magi_expert_dic
  
        for expert_idx in range(self.num_experts*self.world_size):
            if (layer,expert_idx) in self.magi_expert_dic:
                if self.magi_expert_dic[(layer,expert_idx)]>self.magi_expert_dic[(layer,global_expert_idx)]:
                    self.magi_expert_dic[(layer,expert_idx)]-=1
        del self.magi_expert_dic[(layer,global_expert_idx)]

        log.send_del_log(f'rank {self.rank} del expert {global_expert_idx} (origin on rank {global_expert_idx//self.num_experts}) on layer {layer}')

