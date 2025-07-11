import torch
import time
from collections import deque
import torch.distributed as dist
from magi import experts as magi_experts
from magi import expert_utils
from magi import log
from magi import policy

class magi_runtime():
    def __init__(self,args):

        # policy_interval<model_keep_time<2*policy_interval
        self.model_keep_time=13
        self.window_size=10
        self.policy_interval=10
        self.proxy_expert_nums=args.num_layers*args.fmoe_num_experts*args.data_parallel_size//4
        
        self.model=args.magi_model
        self.d_model = args.hidden_size
        self.num_layers = args.num_layers
        self.num_experts = args.fmoe_num_experts
        self.world_size = args.data_parallel_size
        self.rank=args.rank
        self.topk=args.top_k
        self.global_batch_size=args.global_batch_size
        self.total_input_size=args.seq_length*args.micro_batch_size*args.top_k*args.data_parallel_size
        self.seq_length=args.seq_length
        self.gate=args.balance_strategy

        self.magi_profile_flag=args.magi_profile_flag
        self.magi_redirect=args.magi_token_redirect_flag
        self.magi_no_policy=args.magi_no_policy

        # pl means per layer
        # self.pl_local_token_count=[None] * self.num_layers
        # self.pl_global_token_count=[None] * self.num_layers
        # self.pl_all_rank_global_token_count=[None] * self.num_layers
        self.pl_record_fowd_time={'stime':[0]* self.num_layers,
                                      'ctime':[0]* self.num_layers,
                                      'ctime_wait':[0]* self.num_layers,
                                      'rtime':[0]* self.num_layers,
                                      'rtime_wait':[0]* self.num_layers,
                                      'magi_stime':[0]* self.num_layers,
                                      'magi_ctime':[0]* self.num_layers,
                                      'magi_ctime_wait':[0]* self.num_layers,
                                      'keep_ctime':[0]* self.num_layers}
        self.pl_record_bawd_time={'stime':[0]* self.num_layers,
                                      'ctime':[0]* self.num_layers,
                                      'ctime_wait':[0]* self.num_layers,
                                      'rtime':[0]* self.num_layers,
                                      'rtime_wait':[0]* self.num_layers,
                                      'magi_ctime':[0]* self.num_layers,
                                      'magi_reduce':[0]* self.num_layers,
                                      'keep_ctime':[0]* self.num_layers,
                                      'keep_reduce':[0]* self.num_layers,
                                      'set_gradients':[0]* self.num_layers}
        
        self.pl_send=torch.zeros(self.num_layers,self.world_size*self.num_experts, dtype=torch.bool)
        # [expert_idx*self.world_size+rank_idx] means rank_idx shoud receive expert_idx
        self.pl_receive=torch.zeros(self.num_layers,self.world_size*self.num_experts*self.world_size, dtype=torch.bool)
        self.global_pl_keep=[torch.zeros(self.num_layers, self.world_size*self.num_experts, dtype=torch.int32) for _ in range(self.world_size)]

        # self.local_token_deque=deque(maxlen=self.window_size)
        # self.global_token_deque=deque(maxlen=self.window_size)
        self.all_rank_global_token_deque=deque(maxlen=self.window_size)
        self.all_rank_global_token_deque.appendleft([None] * self.num_layers)

        self.eval=False
        self.itr=0
        self.layer=0
        
        if args.janus:
            self.init_janus()
        policy.init_policy(self.world_size,self.num_experts,self.num_layers,self.model_keep_time,self.proxy_expert_nums)
        log.init_log(self)
        self.magi_expert=magi_experts.magi_expert(self)
        self.janus=args.janus

    def set_eval(self,eval_flag):
        self._init_send_receive_models()
        self.eval=eval_flag

    def init_janus(self):
        self.window_size=1
        self.model_keep_time=1
        self.policy_interval=1
        self.proxy_expert_nums=self.num_layers*self.num_experts*self.world_size
    # def record_local_expert_count(self,local_expert_count):
    #     if not self.eval:
    #         self.pl_local_token_count[self.layer]=local_expert_count

    # def record_global_expert_count(self,global_expert_count):
    #     if not self.eval:
    #         self.pl_global_token_count[self.layer]=global_expert_count
    #         log.save_global_token_log(self.gate,self.layer,self.itr,global_expert_count)
    
    def record_all_rank_global_expert_count(self,all_rank_global_token_count):
        if not self.eval:
            self.all_rank_global_token_deque[0][self.layer]=all_rank_global_token_count
            log.save_global_token_log(self,all_rank_global_token_count)

    def record_fwd_layer_time(self,stime,ctime,ctime_wait,rtime,rtime_wait,magi_stime,magi_ctime,magi_ctime_wait,keep_ctime):
        if not self.eval:
            self.pl_record_fowd_time['stime'][self.layer]=stime
            self.pl_record_fowd_time['ctime'][self.layer]=ctime
            self.pl_record_fowd_time['ctime_wait'][self.layer]=ctime_wait
            self.pl_record_fowd_time['rtime'][self.layer]=rtime
            self.pl_record_fowd_time['rtime_wait'][self.layer]=rtime_wait
            self.pl_record_fowd_time['magi_stime'][self.layer]=magi_stime
            self.pl_record_fowd_time['magi_ctime'][self.layer]=magi_ctime
            self.pl_record_fowd_time['magi_ctime_wait'][self.layer]=magi_ctime_wait
            self.pl_record_fowd_time['keep_ctime'][self.layer]=keep_ctime
    
    def record_bwd_layer_time(self,stime,ctime,ctime_wait,rtime,rtime_wait,magi_ctime,magi_reduce,keep_ctime,keep_reduce,set_gradients):
        if not self.eval:
            self.pl_record_bawd_time['stime'][self.layer]=stime
            self.pl_record_bawd_time['ctime'][self.layer]=ctime
            self.pl_record_bawd_time['ctime_wait'][self.layer]=ctime_wait
            self.pl_record_bawd_time['rtime'][self.layer]=rtime
            self.pl_record_bawd_time['rtime_wait'][self.layer]=rtime_wait
            self.pl_record_bawd_time['magi_ctime'][self.layer]=magi_ctime
            self.pl_record_bawd_time['magi_reduce'][self.layer]=magi_reduce
            self.pl_record_bawd_time['keep_ctime'][self.layer]=keep_ctime
            self.pl_record_bawd_time['keep_reduce'][self.layer]=keep_reduce
            self.pl_record_bawd_time['set_gradients'][self.layer]=set_gradients

    def get_send_models(self):
        return self.pl_send[self.layer]

    def get_receive_models(self):
        return self.pl_receive[self.layer]
    
    def get_keep_models(self):
        return torch.cat(self.global_pl_keep,dim=1)[self.layer]
    
    def get_global_keep_models_nums(self,layer,expert_idx):
        # origin expert rank is not cnt in keep_models
        cnt=1
        for rank_idx in range(self.world_size):
            if self.global_pl_keep[rank_idx][layer][expert_idx]>0:
                cnt+=1
        return cnt
    
    def get_redirect_models(self):
        re_send=torch.full(size=tuple([self.num_experts*self.world_size]),fill_value=-1,dtype=torch.int32)
        re_receive=torch.zeros(self.num_experts*self.world_size*self.world_size, dtype=torch.bool)
        re_unreceive=torch.zeros(self.num_experts*self.world_size*self.world_size, dtype=torch.bool)
        # re_send: change expert_idx's token target from global origin expert to global magi expert
        # re_receive: expert_idx's token receive for local magi expert
        # re_unreceive: expert_idx's token unreceive for local origin expert

        if not self.magi_redirect:
            return re_send,re_receive,re_unreceive
        
        local_keep_models=self.global_pl_keep[self.rank][self.layer]
        for expert_idx in range(self.world_size*self.num_experts):
            origin_rank=expert_idx//self.num_experts
            global_keep_models_nums=self.get_global_keep_models_nums(self.layer,expert_idx)
            keep_rank_interval=self.world_size//global_keep_models_nums
            
            # re_send
            if global_keep_models_nums>1:
                # expert_idx exist global magi models
                send_rank=(origin_rank+keep_rank_interval*\
                                       (self.rank//keep_rank_interval-origin_rank//keep_rank_interval))\
                                        %self.world_size
                # don't send local token to local magi/origin expert
                # don't send local token to global origin expert
                if (send_rank!=self.rank and send_rank!=origin_rank):
                    re_send[expert_idx]=send_rank
            
            # re_receive
            if local_keep_models[expert_idx]>0: 
                # local has expert_idx's magi expert
                keep_rank=self.rank//keep_rank_interval*keep_rank_interval
                re_receive[expert_idx*self.world_size+keep_rank:expert_idx*self.world_size+keep_rank+keep_rank_interval]=True
                # don't receive local token for local magi/origin expert
                re_receive[expert_idx*self.world_size+self.rank]=False
                # don't receive global token for local origin expert
                # re_receive[expert_idx*self.world_size+origin_rank]=0
            
            # re_unreceive
            if global_keep_models_nums>1 and origin_rank==self.rank:
                # expert_idx exist global magi models and local has origin expert
                keep_rank=self.rank//keep_rank_interval*keep_rank_interval
                re_unreceive[expert_idx*self.world_size:expert_idx*self.world_size+self.world_size]=True
                re_unreceive[expert_idx*self.world_size+keep_rank:expert_idx*self.world_size+keep_rank+keep_rank_interval]=False
                # receive global token for local origin expert
                offset=self.rank%keep_rank_interval
                for i in range(expert_idx*self.world_size+offset,expert_idx*self.world_size+offset+self.world_size,keep_rank_interval):
                    re_unreceive[i]=False

        return re_send,re_receive,re_unreceive
    
    def get_redirect_expert_count(self,re_receive):
        if not self.magi_redirect:
            return torch.tensor([],dtype=torch.long)
        redirect_expert_count = list()
        for i in range(len(re_receive)):
            if (re_receive[i]):
                recv_from_rank=i%self.world_size
                recv_from_expert_idx=i//self.world_size
                
                offset=recv_from_rank*self.num_experts+(recv_from_expert_idx%self.num_experts)
                redirect_expert_count.append(self.all_rank_global_token_deque[0][self.layer][recv_from_expert_idx//self.num_experts*self.world_size*self.num_experts+offset].item())
        return torch.tensor(redirect_expert_count,dtype=torch.long)
    
    # def get_redirect_token_receive_from_which_rank(self,layer=-1):
    #     if layer==-1:
    #         layer=self.layer
    #     res=torch.zeros(self.num_experts*self.world_size*self.world_size, dtype=torch.bool)
    #     if not self.magi_redirect:
    #         return res
        
    #     keep_models=self.global_pl_keep[self.rank][layer]
    #     for expert_idx in range(self.world_size*self.num_experts):
    #         keep_models_nums=self.get_global_keep_models_nums(layer,expert_idx)
    #         if keep_models[expert_idx]>0: 
    #             # this rank has expert_idx's keep_models
    #             keep_rank_interval=self.world_size//keep_models_nums
    #             keep_rank=self.rank//keep_rank_interval*keep_rank_interval
    #             res[expert_idx*self.world_size+keep_rank:expert_idx*self.world_size+keep_rank+keep_rank_interval]=True
    #     # res[expert_idx*self.world_size+rank_idx]=True means this rank should receive expert_idx's token from rank_idx
    #     return res
                    
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

    # using local_token/global_token to calculate origin_token/receive_token
    def lg_token_to_or_token(self,layer,itr_cnt=0):
        origin_token=list()
        receive_token=list()
        if itr_cnt==0:
            itr_cnt=self.window_size
        all_rank_global_token_count_tensor=self.all_rank_global_token_deque[0][layer]
        
        for i in range(1,min(itr_cnt,len(self.all_rank_global_token_deque))):
            all_rank_global_token_count_tensor=torch.add(all_rank_global_token_count_tensor,self.all_rank_global_token_deque[i][layer])
    
        for expert_idx in range(self.world_size*self.num_experts):
            rank=expert_idx//self.num_experts
            global_token_count=all_rank_global_token_count_tensor[rank*self.num_experts*self.world_size:(rank+1)*self.num_experts*self.world_size]
            # token needn't to send to other workers
            origin_token.append(global_token_count[rank*self.num_experts+expert_idx%self.num_experts].item())
            # token need to receive from other workers
            receive_token.append(sum(global_token_count[(expert_idx%self.num_experts)::self.num_experts]).item()-origin_token[-1])
        
        log.save_or_token_log(self,receive_token,origin_token)
      
        return origin_token,receive_token
    
    def _init_send_receive_models(self):
        self.pl_send.zero_()
        self.pl_receive.zero_()

    def _receive_or_not(self,layer_idx,expert_idx,rank_idx):
        # should this rank receive this expert
        return self.pl_send[layer_idx][expert_idx] and self.pl_receive[layer_idx][expert_idx*self.world_size+rank_idx]
    
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
        send_tensor = self.global_pl_keep[self.rank].cuda(torch.cuda.current_device()).contiguous()
        receive_tensor_list=[torch.zeros(self.num_layers,self.world_size*self.num_experts, dtype=torch.int32,device=torch.cuda.current_device()).contiguous() for _ in range(self.world_size)]
        
        dist.all_gather(receive_tensor_list,send_tensor)

        self.global_pl_keep=[tensor.cpu() for tensor in receive_tensor_list]

    def reset_layer(self):
        self.layer=0

    def next_layer(self):
        log.print_time(self.pl_record_fowd_time,fowd=True,layer=self.layer)
        self.lg_token_to_or_token(self.layer,itr_cnt=1)
        self.layer+=1
    
    def pre_layer(self):
        self.layer-=1
        log.print_time(self.pl_record_bawd_time,fowd=False,layer=self.layer)
            
    def next_itr(self):
        self.itr+=1
        # log.print_time(self.pl_record_fowd_time,fowd=True)
        # log.print_time(self.pl_record_bawd_time,fowd=False)

        # update keep_models
        self._cnt_down_keep_models()

        if self.itr%self.policy_interval==0:
            self.pl_send,self.pl_receive,self.global_pl_keep=policy.using_policy(self)
            log.print_policy_tensor(f'rank:{self.rank} pl_send:{self.pl_send} pl_receive:{self.pl_receive} global_pl_keep:{self.global_pl_keep}')
        elif self.itr%self.policy_interval==1:
            self._init_send_receive_models()
        log.save_keep_log(self,self.global_pl_keep)
        
        self.all_rank_global_token_deque.appendleft([None] * self.num_layers)
        self.reset_layer()
        