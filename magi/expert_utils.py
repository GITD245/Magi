import torch


class magi_expert:
    def __init__(self,args):
        self.magi_expert_dic={}
        self.magi_profile=args.magi_profile
        self.rank=args.rank
        self.num_experts=args.fmoe_num_experts

    # def _create_magi_expert_buffer(self,input_buffer,expert_size,global_expert_idx):
    #     layer=self.magi_profile.get_layer()
    #     new_expert_buffer=input_buffer.new_empty(expert_size);
    #     self.magi_expert_dic[(layer,global_expert_idx)]=new_expert_buffer
    #     return new_expert_buffer
    
    def get_magi_expert(self,out,global_expert_idx):
        print(self.magi_expert_dic)
        layer=self.magi_profile.get_layer()
        return self.magi_expert_dic[(layer,global_expert_idx)]
    
    def registe_magi_expert(self,out,global_expert_idx,e,get_params=False):
        # out=self._create_magi_expert_buffer(input_buffer,expert_size,global_expert_idx)

        if get_params:
            e = e[global_expert_idx%self.num_experts]
            offset = 0
            for n, p in e.named_parameters():
                seg = out[offset:offset + p.numel()]
                offset += p.numel()
                seg.copy_(p.data.flatten())
        
        self.magi_expert_dic[(self.magi_profile.get_layer(),global_expert_idx)]=out
        # self._print_rank_0(self.magi_expert_dic)


    def _print_rank_0(self,str):
        if self.rank==0:
            print(str)