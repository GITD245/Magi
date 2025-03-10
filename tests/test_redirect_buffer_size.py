import torch
magi_redirect=0
re_receive=torch.tensor([0,0,0,0,
                         0,0,0,0,
                         0,0,1,0,
                         0,0,0,0,
                         0,0,0,0,
                         0,0,0,0,
                         0,0,0,0,
                         0,0,0,0],dtype=torch.bool)
world_size=4
num_experts=2
pl_all_rank_global_token_count=torch.tensor([0,1,2,3,4,5,6,7,
                                             8,9,10,11,12,13,14,15,
                                             16,17,18,19,20,21,22,23,
                                             24,25,26,27,28,29,30,31],dtype=torch.int32)
                                


def get_redirect_global_buffer_size(re_receive):
        redirect_expert_count = list()
        if not magi_redirect:
            return torch.tensor([],dtype=torch.int32)
        for i in range(len(re_receive)):
            if (re_receive[i]):
                recv_from_rank=i%world_size
                recv_from_expert_idx=i//world_size
                print(f'recv_from_rank:{recv_from_rank} recv_from_expert_idx:{recv_from_expert_idx}')
                offset=recv_from_rank*num_experts+(recv_from_expert_idx%num_experts)
                redirect_expert_count.append(pl_all_rank_global_token_count[recv_from_expert_idx//num_experts*world_size*num_experts+offset].item())
        return torch.tensor(redirect_expert_count,dtype=torch.int32)

print(get_redirect_global_buffer_size(re_receive))