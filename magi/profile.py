import torch

class magi_profile():
    def __init__(self,args,window_size=10):
        self.num_layers = args.num_layers
        self.num_experts = args.fmoe_num_experts
        self.rank=args.rank
        self.window_size = window_size

        self.per_layer_local_token_count=[[] for _ in range(self.num_layers)]
        self.per_layer_global_token_count=[[] for _ in range(self.num_layers)]

        if args.magi_profile_flag:
            self.record_time=torch.zeros(3, dtype=torch.float32)

        self.external_token=0
        self.eval=False
        self.itr=0
        self.layer_idx=0
    def set_eval(self,eval_flag):
        self.eval=eval_flag


    def record_local_expert_count(self,local_expert_count):
        if not self.eval:
            self.per_layer_local_token_count[self.layer_idx]=local_expert_count

    def record_global_expert_count(self,global_expert_count):
        if not self.eval:
            self.per_layer_global_token_count[self.layer_idx]=global_expert_count

    def record_layer_time(self,stime,ctime,rtime):
        if not self.eval:
            print(f"rank {self.rank} stime: {stime}, ctime: {ctime}, rtime: {rtime}")

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
    