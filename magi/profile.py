import torch
from collections import deque
import torch.distributed as dist

class magi_profile():
    def __init__(self,args,window_size=10):
        self.print_time=False
        self.print_token=False
        self.save_global_token_log=False

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


        self.external_token=0
        self.eval=False
        self.itr=1
        self.layer_idx=0

    def _lg_token_to_or_token(self,layer,itr=0):
        origin_token={}
        recive_token={}

        for expert_idx in range(self.rank*self.num_experts,self.rank*self.num_experts+self.num_experts):
            # token needn't to be sent to other workers
            origin_token[expert_idx]=self.global_token_deque[itr][layer][self.rank*self.num_experts+expert_idx%self.num_experts].item()
            # token need to be recived from other workers
            recive_token[expert_idx]=sum(self.global_token_deque[itr][layer][(expert_idx%self.num_experts)::self.num_experts]).item()-origin_token[expert_idx]
        if self.print_token:
            print(f"itr :{self.itr} layer:{layer} recive_token:{recive_token} origin_token:{origin_token}")
        
        return origin_token,recive_token

    def set_eval(self,eval_flag):
        self.eval=eval_flag

    def record_local_expert_count(self,local_expert_count):
        if not self.eval:
            self.per_itr_local_token_count[self.layer_idx]=local_expert_count

    def record_global_expert_count(self,global_expert_count):
        if not self.eval:
            self.per_itr_global_token_count[self.layer_idx]=global_expert_count
        
        if self.save_global_token_log:
            with open(f"log/{self.gate}_{self.rank}_global_token_count",'a') as f:
                f.write(f"layer:{self.layer_idx} itr:{self.itr} global_expert_count:{global_expert_count}\n")

    def record_layer_time(self,stime,ctime,rtime,shadow_stime,shadow_ctime):
        if not self.eval:
            self.per_itr_record_time['stime'][self.layer_idx]=stime
            self.per_itr_record_time['ctime'][self.layer_idx]=ctime
            self.per_itr_record_time['rtime'][self.layer_idx]=rtime
            self.per_itr_record_time['shadow_stime'][self.layer_idx]=shadow_stime
            self.per_itr_record_time['shadow_ctime'][self.layer_idx]=shadow_ctime
    
    def get_layer(self):
        return self.layer_idx
    
    def get_itr(self):
        return self.itr
    
    def get_world_size(self):
        return self.world_size
    
    def get_num_experts(self):
        return self.num_experts
    
    def next_layer(self):
        if not self.eval:
            self.layer_idx+=1
            
    def next_itr(self):
             
        self.local_token_deque.appendleft([value for value in self.per_itr_local_token_count])
        self.global_token_deque.appendleft([value for value in self.per_itr_global_token_count])

        self._lg_token_to_or_token(layer=5)

        if self.magi_profile_flag and self.print_time:
            stime=sum(self.per_itr_record_time['stime'])
            ctime=sum(self.per_itr_record_time['ctime'])
            rtime=sum(self.per_itr_record_time['rtime'])
            shadow_stime=sum(self.per_itr_record_time['shadow_stime'])
            shadow_ctime=sum(self.per_itr_record_time['shadow_ctime'])
            # stime=self.per_itr_record_time['stime'][5]
            # ctime=self.per_itr_record_time['ctime'][5]
            # rtime=self.per_itr_record_time['rtime'][5]
            # shadow_stime=self.per_itr_record_time['shadow_stime'][5]
            # shadow_ctime=self.per_itr_record_time['shadow_ctime'][5]

            print(f"rank:{self.rank} layer:all stime:{stime:.2f} ctime:{ctime:.2f} rtime:{rtime:.2f} shadow_stime:{shadow_stime:.2f} shadow_ctime:{shadow_ctime:.2f}\n")

        self.itr+=1
        self.layer_idx=0

    def _print_rank_0(self,str):
        if self.rank==0:
            print(str)

if __name__ == "__main__":
    pass