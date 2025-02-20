import torch
world_size=8
num_experts=4
origin_expert=8
keep_models=torch.zeros(8)
keep_models[0]=1
rank=1

def get_global_keep_models_nums():
    # cnt=0
    # for i in keep_models:
    #     if i>0:
    #         cnt+1
    # if cnt==0:
    #     return 1
    # return cnt
    return 3

def token_send_to_which_rank():
    res=torch.full(size=tuple([world_size]), fill_value=-1,dtype=torch.int)
    for rank in range(world_size):
        origin_send_rank=origin_expert//num_experts
        # if origin_send_rank==rank:
        #     # local expert needn't to send
        #     continue
        keep_model_nums=get_global_keep_models_nums()
        keep_rank_interval=world_size//keep_model_nums
        if  keep_model_nums==1:
            # no proxy expert
            continue
        else:
            send_rank=(origin_send_rank+keep_rank_interval*\
                                    (rank//keep_rank_interval-origin_send_rank//keep_rank_interval))\
                                    %world_size
            # if send_rank==origin_send_rank:
            #     continue
            # else:
            #     res[rank]=send_rank
            #     continue
            res[rank]=send_rank
    print(res)

def token_receive_from_which_rank():
    res=torch.zeros(num_experts*world_size*world_size, dtype=torch.bool)
    if not keep_models.any():
        print(res)
        return
    for expert_idx in range(world_size*num_experts):
        if keep_models[expert_idx]>0:
            keep_model_nums=get_global_keep_models_nums()
            if keep_model_nums>1:
                keep_rank_interval=world_size//keep_model_nums
                keep_rank=rank//keep_rank_interval*keep_rank_interval
                res[expert_idx*world_size+keep_rank:expert_idx*world_size+keep_rank+keep_rank_interval]=True
    print(res)

token_send_to_which_rank()
# token_receive_from_which_rank()