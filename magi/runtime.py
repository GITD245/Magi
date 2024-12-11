import torch
from collections import deque
import torch.distributed as dist
from magi import expert_utils
from magi import log

class magi_runtime():
    def __init__(self,args,window_size=10):

        self.num_layers = args.num_layers
        self.num_experts = args.fmoe_num_experts
        self.world_size = args.data_parallel_size
        self.rank=args.rank
        self.window_size = window_size
        self.total_input_size=args.seq_length*args.micro_batch_size*args.top_k*args.data_parallel_size
        self.magi_profile_flag=args.magi_profile_flag
        self.gate=args.balance_strategy

        self.per_itr_local_token_count=[None] * self.num_layers
        self.per_itr_global_token_count=[None] * self.num_layers
        self.per_itr_record_time={'stime':[0]* self.num_layers,
                                      'ctime':[0]* self.num_layers,
                                      'rtime':[0]* self.num_layers,
                                      'shadow_stime':[0]* self.num_layers,
                                      'shadow_ctime':[0]* self.num_layers}

        self.local_token_deque=deque(maxlen=self.window_size)
        self.global_token_deque=deque(maxlen=self.window_size)


        self.eval=False
        self.itr=1
        self.layer=0

        log.set_rank(args.rank)
        self.magi_expert=expert_utils.magi_expert(self)

    def _lg_token_to_or_token(self,layer=0,itr=0):
        origin_token={}
        recive_token={}

        for expert_idx in range(self.rank*self.num_experts,self.rank*self.num_experts+self.num_experts):
            # token needn't to be sent to other workers
            origin_token[expert_idx]=self.global_token_deque[itr][layer][self.rank*self.num_experts+expert_idx%self.num_experts].item()
            # token need to be recived from other workers
            recive_token[expert_idx]=sum(self.global_token_deque[itr][layer][(expert_idx%self.num_experts)::self.num_experts]).item()-origin_token[expert_idx]
        
        log.print_token(self.itr,layer,recive_token,origin_token)
      
        return origin_token,recive_token

    def set_eval(self,eval_flag):
        self.eval=eval_flag

    def record_local_expert_count(self,local_expert_count):
        if not self.eval:
            self.per_itr_local_token_count[self.layer]=local_expert_count

    def record_global_expert_count(self,global_expert_count):
        if not self.eval:
            self.per_itr_global_token_count[self.layer]=global_expert_count

            log.save_global_token_log(self.gate,self.layer,self.itr,global_expert_count)


    def record_layer_time(self,stime,ctime,rtime,shadow_stime,shadow_ctime):
        if not self.eval:
            self.per_itr_record_time['stime'][self.layer]=stime
            self.per_itr_record_time['ctime'][self.layer]=ctime
            self.per_itr_record_time['rtime'][self.layer]=rtime
            self.per_itr_record_time['shadow_stime'][self.layer]=shadow_stime
            self.per_itr_record_time['shadow_ctime'][self.layer]=shadow_ctime
    
    def get_layer(self):
        return self.layer

    def get_rank(self):
        return self.rank
    
    def get_itr(self):
        return self.itr
    
    def get_world_size(self):
        return self.world_size
    
    def get_num_experts(self):
        return self.num_experts
    
    def next_layer(self):
        if not self.eval:
            self.layer+=1
            
    def next_itr(self):
             
        self.local_token_deque.appendleft([value for value in self.per_itr_local_token_count])
        self.global_token_deque.appendleft([value for value in self.per_itr_global_token_count])

        self._lg_token_to_or_token()

        if self.magi_profile_flag:
            log.print_time(self.per_itr_record_time)

        self.itr+=1
        self.layer=0
