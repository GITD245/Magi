import torch
from collections import deque
import torch.distributed as dist
from magi import experts as magi_experts
from magi import expert_utils
from magi import log
from magi import policy

class magi_runtime():
    def __init__(self,args,window_size=10):

        self.model_keep_time=10
        self.policy_interval=5
        
        self.d_model = args.hidden_size
        self.num_layers = args.num_layers
        self.num_experts = args.fmoe_num_experts
        self.world_size = args.data_parallel_size
        self.rank=args.rank
        self.window_size = window_size
        self.total_input_size=args.seq_length*args.micro_batch_size*args.top_k*args.data_parallel_size
        self.magi_profile_flag=args.magi_profile_flag
        self.gate=args.balance_strategy

        # pl means per layer
        self.pl_local_token_count=[None] * self.num_layers
        self.pl_global_token_count=[None] * self.num_layers
        self.pl_record_time={'stime':[0]* self.num_layers,
                                      'ctime':[0]* self.num_layers,
                                      'ctime_wait':[0]* self.num_layers,
                                      'rtime':[0]* self.num_layers,
                                      'rtime_wait':[0]* self.num_layers,
                                      'magi_stime':[0]* self.num_layers,
                                      'magi_ctime':[0]* self.num_layers,
                                      'magi_ctime_wait':[0]* self.num_layers,
                                      'keep_ctime':[0]* self.num_layers}
        
        self.pl_send=torch.zeros(self.num_layers,self.world_size*self.num_experts, dtype=torch.bool)
        # [i*self.world_size+rank] means the expert_idx i shoud be recived from which rank
        self.pl_receive=torch.zeros(self.num_layers,self.world_size*self.num_experts*self.world_size, dtype=torch.bool)
        self.global_pl_keep=[torch.zeros(self.num_layers, self.world_size*self.num_experts, dtype=torch.int,device=torch.cuda.current_device()).contiguous() for _ in range(self.world_size)]

        self.local_token_deque=deque(maxlen=self.window_size)
        self.global_token_deque=deque(maxlen=self.window_size)

        self.eval=False
        self.itr=1
        self.layer=0
        policy.init_policy(self.world_size,self.num_experts,self.num_layers)
        log.init_log(args.rank,args.magi_profile_flag)
        self.magi_expert=magi_experts.magi_expert(self)

    def set_eval(self,eval_flag):
        self._init_send_receive_models()
        self.eval=eval_flag

    def record_local_expert_count(self,local_expert_count):
        if not self.eval:
            self.pl_local_token_count[self.layer]=local_expert_count

    def record_global_expert_count(self,global_expert_count):
        if not self.eval:
            self.pl_global_token_count[self.layer]=global_expert_count
            log.save_global_token_log(self.gate,self.layer,self.itr,global_expert_count)

    def record_layer_time(self,stime,ctime,ctime_wait,rtime,rtime_wait,magi_stime,magi_ctime,magi_ctime_wait,keep_ctime):
        if not self.eval:
            self.pl_record_time['stime'][self.layer]=stime
            self.pl_record_time['ctime'][self.layer]=ctime
            self.pl_record_time['ctime_wait'][self.layer]=ctime_wait
            self.pl_record_time['rtime'][self.layer]=rtime
            self.pl_record_time['rtime_wait'][self.layer]=rtime_wait
            self.pl_record_time['magi_stime'][self.layer]=magi_stime
            self.pl_record_time['magi_ctime'][self.layer]=magi_ctime
            self.pl_record_time['magi_ctime_wait'][self.layer]=magi_ctime_wait
            self.pl_record_time['keep_ctime'][self.layer]=keep_ctime

    def get_send_models(self,layer=-1):
        if layer==-1:
            layer=self.layer
        return self.pl_send[layer]

    def get_receive_models(self,layer=-1):
        if layer==-1:
            layer=self.layer
        return self.pl_receive[layer]
    
    def get_keep_models(self,layer=-1):
        if layer==-1:
            layer=self.layer
        return torch.cat(self.global_pl_keep,dim=1)[layer].cpu()
    
    # def is_magi_expert_exist(self,flag_buf,rank_idx,expert_idx,layer=-1):
    #     if layer==-1:
    #         layer=self.layer
    #     if self.global_pl_keep[rank_idx][layer][expert_idx]>0:
    #         flag_buf[0]=True
    #     else:
    #         flag_buf[0]=False
    
    # def is_global_magi_expert_exist(self,flag_buf,expert_idx,layer=-1):
    #     if layer==-1:
    #         layer=self.layer
    #     cnt=0
    #     for rank_idx in range(self.world_size):
    #         cnt+=self.global_pl_keep[rank_idx][layer][expert_idx]
    #     if cnt>0:
    #         flag_buf[0]=True
    #     else:
    #         flag_buf[0]=False
        
    def _init_send_receive_models(self):
        self.pl_send.zero_()
        self.pl_receive.zero_()

    def _receive_or_not(self,layer_idx,expert_idx,rank_idx):
        # should this rank receive this expert
        return self.pl_send[layer_idx][expert_idx] and self.pl_receive[layer_idx][expert_idx*self.world_size+rank_idx]

    def _lg_token_to_or_token(self,layer=0,itr=0):
        origin_token=dict()
        recive_token=dict()

        for expert_idx in range(self.rank*self.num_experts,self.rank*self.num_experts+self.num_experts):
            # token needn't to be sent to other workers
            origin_token[expert_idx]=self.global_token_deque[itr][layer][self.rank*self.num_experts+expert_idx%self.num_experts].item()
            # token need to be recived from other workers
            recive_token[expert_idx]=sum(self.global_token_deque[itr][layer][(expert_idx%self.num_experts)::self.num_experts]).item()-origin_token[expert_idx]
        
        log.print_token(self.itr,layer,recive_token,origin_token)
      
        return origin_token,recive_token
    
    def _cnt_down_keep_models(self):
        for layer_idx in range(self.num_layers):
            for expert_idx in range(self.world_size*self.num_experts):
                # cnt down keep models
                if self.global_pl_keep[self.rank][layer_idx][expert_idx]>0:
                    if self.global_pl_keep[self.rank][layer_idx][expert_idx]==1:
                        # del model
                        self.magi_expert.del_magi_expert(layer_idx,expert_idx)
                    self.global_pl_keep[self.rank][layer_idx][expert_idx]-=1
                # record send receive models in keep models
                if self._receive_or_not(layer_idx,expert_idx,self.rank):
                    #MAGI_TODO: change keep time?
                    # self rank should not keep self magi_expert
                    if expert_idx//self.num_experts!=self.rank:
                        self.global_pl_keep[self.rank][layer_idx][expert_idx]+=self.model_keep_time
        self._all_gather_keep_models()

    # def _cnt_down_keep_models(self):
    #     for layer_idx in range(self.num_layers):
    #         for expert_idx in range(self.world_size*self.num_experts):
    #             for rank_idx in range(self.world_size):
    #                 # cnt down keep models
    #                 if self.global_pl_keep[rank_idx][layer_idx][expert_idx]>0:
    #                     if self.rank==rank_idx and self.global_pl_keep[self.rank][layer_idx][expert_idx]==1:
    #                         # del model
    #                         self.magi_expert.del_magi_expert(layer_idx,expert_idx)
    #                     self.global_pl_keep[rank_idx][layer_idx][expert_idx]-=1
    #                 # record send receive models in keep models
    #                 if self._receive_or_not(layer_idx,expert_idx,rank_idx):
    #                     #MAGI_TODO: change keep time?
    #                     # self rank should not keep self magi_expert
    #                     if expert_idx//self.num_experts!=rank_idx:
    #                         self.global_pl_keep[rank_idx][layer_idx][expert_idx]+=self.model_keep_time


    def _all_gather_keep_models(self):
        send_tensor = self.global_pl_keep[self.rank].contiguous()
        receive_tensor_list=[torch.zeros(self.num_layers,self.world_size*self.num_experts, dtype=torch.int,device=torch.cuda.current_device()).contiguous() for _ in range(self.world_size)]
        
        dist.all_gather(receive_tensor_list,send_tensor)
        self.global_pl_keep=receive_tensor_list

    def next_layer(self):
        # log._print(f'runtime_layer:{self.layer}')
        if not self.eval:
            self.layer+=1
            
    def next_itr(self):
        self.local_token_deque.appendleft([value for value in self.pl_local_token_count])
        self.global_token_deque.appendleft([value for value in self.pl_global_token_count])
        log.print_time(self.pl_record_time)

        self._lg_token_to_or_token()
        # MAGI_TODO: Reduce or_token to rank0 ?

        # update keep_models
        self._cnt_down_keep_models()
        
        if self.itr%self.policy_interval==0:
            self.pl_send,self.pl_receive,self.global_pl_keep=policy.using_policy(self)
            log.print_policy_tensor(f'rank:{self.rank} pl_send:{self.pl_send} pl_receive:{self.pl_receive} global_pl_keep:{self.global_pl_keep}')
        elif self.itr%self.policy_interval==1:
            self._init_send_receive_models()
            
        self.itr+=1
        self.layer=0
        