import torch
import torch.distributed as dist
import time
from magi import log

WORLD_SIZE=0
NUM_EXPERTS=0
NUM_LAYERS=0
MODEL_KEEP_TIME=0

def init_policy(world_size,num_experts,num_layers,model_keep_time):
    global WORLD_SIZE
    global NUM_EXPERTS
    global NUM_LAYERS
    global MODEL_KEEP_TIME
    WORLD_SIZE=world_size
    NUM_EXPERTS=num_experts
    NUM_LAYERS=num_layers
    MODEL_KEEP_TIME=model_keep_time

def _receive_or_not(layer_idx,expert_idx,rank_idx,pl_send,pl_receive):
    # should this rank receive this expert
    return pl_send[layer_idx][expert_idx] and pl_receive[layer_idx][expert_idx*WORLD_SIZE+rank_idx]

# def _check_send_receive_keep(model_keep_time,pl_send,pl_receive,global_pl_keep):
#     for layer_idx in range(NUM_LAYERS):
#         for expert_idx in range(WORLD_SIZE*NUM_EXPERTS):
#             for rank_idx in range(WORLD_SIZE):
#                 if _receive_or_not(layer_idx,expert_idx,rank_idx,pl_send,pl_receive):
#                     # same rank should not send and receive self expert
#                     if expert_idx//NUM_EXPERTS==rank_idx:
#                         pl_receive[layer_idx][expert_idx*WORLD_SIZE+rank_idx]=False
#                     # allready have this expert, dont need to receive
#                     elif global_pl_keep[rank_idx][layer_idx][expert_idx]>0:
#                         pl_receive[layer_idx][expert_idx*WORLD_SIZE+rank_idx]=False
#                         # MAGI_TODO:change keep time?
#                         global_pl_keep[rank_idx][layer_idx][expert_idx]+=model_keep_time
#                     else:
#                         log.send_del_log(f'rank {rank_idx} receive expert {expert_idx} (origin on rank {expert_idx//NUM_EXPERTS}) on layer {layer_idx}')
#             # no receive expert, dont need to send
#             if pl_send[layer_idx][expert_idx] and not pl_receive[layer_idx][expert_idx*WORLD_SIZE:expert_idx*WORLD_SIZE+WORLD_SIZE].any():
#                 pl_send[layer_idx][expert_idx]=False

def _check_send_receive_keep(layer_idx,expert_idx,pl_send,pl_receive,global_pl_keep):
    for rank_idx in range(WORLD_SIZE):
        if _receive_or_not(layer_idx,expert_idx,rank_idx,pl_send,pl_receive):
            # same rank should not send and receive self expert
            if expert_idx//NUM_EXPERTS==rank_idx:
                pl_receive[layer_idx][expert_idx*WORLD_SIZE+rank_idx]=False
            # allready have this expert, dont need to receive
            elif global_pl_keep[rank_idx][layer_idx][expert_idx]>0:
                pl_receive[layer_idx][expert_idx*WORLD_SIZE+rank_idx]=False
                # MAGI_TODO:change keep time?
                global_pl_keep[rank_idx][layer_idx][expert_idx]+=MODEL_KEEP_TIME
            else:
                log.send_del_log(f'rank {rank_idx} receive expert {expert_idx} (origin on rank {expert_idx//NUM_EXPERTS}) on layer {layer_idx}')
    # no receive expert, dont need to send
    if pl_send[layer_idx][expert_idx] and not pl_receive[layer_idx][expert_idx*WORLD_SIZE:expert_idx*WORLD_SIZE+WORLD_SIZE].any():
        pl_send[layer_idx][expert_idx]=False

def _update_send_receive_keep_on_all_rank(pl_send,pl_receive,global_pl_keep):
    # broadcast_tensor=torch.cat(global_pl_keep,dim=1).cuda(torch.cuda.current_device())
    # broadcast_tensor=torch.cat([broadcast_tensor,pl_send,pl_receive],dim=1)

    # dist.broadcast(broadcast_tensor, src=0)

    # keep_tensor=broadcast_tensor[:,:WORLD_SIZE*NUM_EXPERTS*WORLD_SIZE]
    # pl_send=broadcast_tensor[:,WORLD_SIZE*NUM_EXPERTS*WORLD_SIZE:WORLD_SIZE*NUM_EXPERTS*WORLD_SIZE+WORLD_SIZE*NUM_EXPERTS].to(torch.bool).cpu()
    # pl_receive=broadcast_tensor[:,WORLD_SIZE*NUM_EXPERTS*WORLD_SIZE+WORLD_SIZE*NUM_EXPERTS:].to(torch.bool).cpu()

    # return pl_send.cpu(),pl_receive.cpu(),list(torch.split(keep_tensor.cpu(),WORLD_SIZE*NUM_EXPERTS,dim=1))

    keep_tensor=torch.cat(global_pl_keep,dim=1).cuda(torch.cuda.current_device())
    
    dist.broadcast(pl_send, src=0)
    dist.broadcast(pl_receive, src=0)
    dist.broadcast(keep_tensor, src=0)

    pl_send=pl_send.cpu()
    pl_receive=pl_receive.cpu()

    return pl_send,pl_receive,list(torch.split(keep_tensor.cpu(),WORLD_SIZE*NUM_EXPERTS,dim=1))

def _set_single_expert(layer,send_expert_idx,receive_expert_idx_list,pl_send,pl_receive,global_pl_keep):
    pass

def _set_double_expert(layer,send_expert_idx,receive_rank_idx_list,pl_send,pl_receive,global_pl_keep):
    pl_send[layer][send_expert_idx]=True
    for rank_idx in receive_rank_idx_list:
        if send_expert_idx//NUM_EXPERTS!=rank_idx:
            pl_receive[layer][send_expert_idx*WORLD_SIZE+rank_idx]=True
    log.send_del_log(f'\nRANK {send_expert_idx//NUM_EXPERTS} send expert {send_expert_idx} to RANK {receive_rank_idx_list} on LAYER {layer}')
    _check_send_receive_keep(layer,send_expert_idx,pl_send,pl_receive,global_pl_keep)

def _set_boradcast_expert(layer,boradcast_expert_idx,pl_send,pl_receive,global_pl_keep):
    pl_send[layer][boradcast_expert_idx]=True
    for rank_idx in range(WORLD_SIZE):
        if boradcast_expert_idx//NUM_EXPERTS!=rank_idx:
            pl_receive[layer][boradcast_expert_idx*WORLD_SIZE+rank_idx]=True
    log.send_del_log(f'\nRANK {boradcast_expert_idx//NUM_EXPERTS} send expert {boradcast_expert_idx} to all rank on LAYER {layer}')
    _check_send_receive_keep(layer,boradcast_expert_idx,pl_send,pl_receive,global_pl_keep)

def _gen_double_receive_rank_idx_list(runtime,layer,rank_idx,expert_idx):
    keep_model_nums=runtime.get_global_keep_models_nums(layer,expert_idx)
    receive_rank_interval=max(WORLD_SIZE//(keep_model_nums*2),1)
    receive_rank_idx_list=[(x+rank_idx)%WORLD_SIZE for x in range(0, WORLD_SIZE) if x % receive_rank_interval== 0]
    return receive_rank_idx_list

def _caculate_P_l(runtime,layer,sorted_layer_receive_token,sorted_layer_expert_idx):
    P_l=list()
    oterh_token_count=sum(sorted_layer_receive_token)
    P_token_count=0
    p_max=-1
    s_l=0
    for i in range(len(sorted_layer_receive_token)):
        P_token_count+=sorted_layer_receive_token[i]
        oterh_token_count-=sorted_layer_receive_token[i]
        if oterh_token_count==0:
            break
        p=(P_token_count/(len(P_l)+1))/(oterh_token_count/(WORLD_SIZE*NUM_EXPERTS-len(P_l)-1))
        if p>=p_max:
            p_max=p
            P_l.append(sorted_layer_expert_idx[i])
            s_l+=runtime.get_global_keep_models_nums(layer,sorted_layer_expert_idx[i])
        else:
            break
    return P_l,s_l

def policy(runtime,pl_send,pl_receive):
    if runtime.magi_policy==0:
        # NO POLICY
        return
    all_receive_token=list()
    pl_all_receive_token=list()
    for i in range(NUM_LAYERS):
        _,receive_token=runtime.lg_token_to_or_token(i)
        all_receive_token.extend(receive_token)
        pl_all_receive_token.append(receive_token)
    sorted_expert_idx = sorted(range(len(all_receive_token)), key=lambda k: all_receive_token[k], reverse=True)

    if runtime.janus or runtime.fastermoe or runtime.magi_policy==1:
        # BASIC POLICY 1 RANKING BROADCAST
        for i in range(runtime.proxy_expert_nums):
            expert_idx=sorted_expert_idx[i]%(WORLD_SIZE*NUM_EXPERTS)
            layer=sorted_expert_idx[i]//(WORLD_SIZE*NUM_EXPERTS)
            _set_boradcast_expert(layer,expert_idx,pl_send,pl_receive,runtime.global_pl_keep)
    elif runtime.magi_policy==2:
        # BASIC POLICY 2 RANKING DOUBLE
        for i in range(runtime.proxy_expert_nums):
            expert_idx=sorted_expert_idx[i]%(WORLD_SIZE*NUM_EXPERTS)
            layer=sorted_expert_idx[i]//(WORLD_SIZE*NUM_EXPERTS)
            rank_idx=expert_idx//NUM_EXPERTS
            receive_rank_idx_list=_gen_double_receive_rank_idx_list(runtime,layer,rank_idx,expert_idx)
            _set_double_expert(layer,expert_idx,receive_rank_idx_list,pl_send,pl_receive,runtime.global_pl_keep)
    elif runtime.magi_policy==3:
        # POLICY 3 POPULARITY
        for layer in range(NUM_LAYERS):
            sorted_layer_receive_token=sorted(pl_all_receive_token[layer], reverse=True)
            sorted_layer_expert_idx=sorted(range(len(pl_all_receive_token[layer])), key=lambda k: pl_all_receive_token[layer][k], reverse=True)
            P_l,s_l=_caculate_P_l(runtime,layer,sorted_layer_receive_token,sorted_layer_expert_idx)
            for expert_idx in P_l:
                rank_idx=expert_idx//NUM_EXPERTS
                receive_rank_idx_list=_gen_double_receive_rank_idx_list(runtime,layer,rank_idx,expert_idx)
                _set_double_expert(layer,expert_idx,receive_rank_idx_list,pl_send,pl_receive,runtime.global_pl_keep)
    else:
        return
        
def using_policy(runtime):
    pl_send=torch.zeros(NUM_LAYERS,WORLD_SIZE*NUM_EXPERTS, dtype=torch.bool,device=torch.cuda.current_device())
    pl_receive=torch.zeros(NUM_LAYERS,WORLD_SIZE*NUM_EXPERTS*WORLD_SIZE, dtype=torch.bool,device=torch.cuda.current_device())

    if runtime.rank==0:
        policy(runtime,pl_send,pl_receive)
        # _check_send_receive_keep(runtime.model_keep_time,pl_send,pl_receive,runtime.global_pl_keep)

    return _update_send_receive_keep_on_all_rank(pl_send,pl_receive,runtime.global_pl_keep)
