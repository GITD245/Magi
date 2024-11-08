import torch

class magi_profile():
    def __init__(self,num_layers,num_experts,rank,window_size=10):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.rank=rank
        self.window_size = window_size

        self.per_layer_local_token_count=[[] for _ in range(num_layers)]
        self.per_layer_global_token_count=[[] for _ in range(num_layers)]

        self.external_token=0
        self.eval=False
        self.itr=0
        self.layer_idx=0
    def set_eval(self,eval_flag):
        self.eval=eval_flag


    def recode_local_expert_count(self,local_expert_count):
        if not self.eval:
            self.per_layer_local_token_count[self.layer_idx]=local_expert_count

    def recode_global_expert_count(self,global_expert_count):
        if not self.eval:
            self.per_layer_global_token_count[self.layer_idx]=global_expert_count

    def recode_time(self):
        pass

    def next_itr(self):
        self.itr+=1
        self.layer_idx=0

    def next_layer(self):
        if not self.eval:
            self.layer_idx+=1
            
    def all_gather(self):
        pass