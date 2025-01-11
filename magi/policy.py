import torch
import torch.distributed as dist
from magi import log

WORLD_SIZE=0
NUM_EXPERTS=0

def init_policy(world_size,num_experts):
    global WORLD_SIZE
    global NUM_EXPERTS
    WORLD_SIZE=world_size
    NUM_EXPERTS=num_experts

def _receive_or_not(pl_send,pl_receive,layer_idx,expert_idx,rank_idx):
    # should this rank receive this expert
    return pl_send[layer_idx][expert_idx] and pl_receive[layer_idx][expert_idx*WORLD_SIZE+rank_idx]

def _check_send_receive_keep(num_layers,model_keep_time,pl_send,pl_receive,global_pl_keep):
    for layer_idx in range(num_layers):
        for expert_idx in range(WORLD_SIZE*NUM_EXPERTS):
            for rank_idx in range(WORLD_SIZE):
                if _receive_or_not(pl_send,pl_receive,layer_idx,expert_idx,rank_idx):
                    # same rank should not send and receive
                    if expert_idx//NUM_EXPERTS==rank_idx:
                        pl_receive[layer_idx][expert_idx*WORLD_SIZE+rank_idx]=False
                    # allready have this expert, dont need to receive
                    elif global_pl_keep[rank_idx][layer_idx][expert_idx]>0:
                        pl_receive[layer_idx][expert_idx*WORLD_SIZE+rank_idx]=False
                        # MAGI_TODO:change keep time?
                        global_pl_keep[rank_idx][layer_idx][expert_idx]+=model_keep_time
                    else:
                        log.profile(f'rank {rank_idx} receive expert {expert_idx} (origin on rank {expert_idx//NUM_EXPERTS}) on layer {layer_idx}')
            # no receive expert, dont need to send
            if not pl_send[layer_idx][expert_idx] and not pl_receive[layer_idx][expert_idx*WORLD_SIZE:expert_idx*WORLD_SIZE+WORLD_SIZE].any():
                pl_send[layer_idx][expert_idx]=False

def _update_send_receive_keep_on_all_rank(pl_send,pl_receive,global_pl_keep):
    keep_tensor=torch.cat(global_pl_keep,dim=1)
    # MAGI_TODO
    pl_send=pl_send.cuda(torch.cuda.current_device())
    pl_receive=pl_receive.cuda(torch.cuda.current_device())
    
    dist.broadcast(pl_send, src=0)
    dist.broadcast(pl_receive, src=0)
    dist.broadcast(keep_tensor, src=0)

    pl_send=pl_send.cpu()
    pl_receive=pl_receive.cpu()

    return pl_send,pl_receive,list(torch.split(keep_tensor,pl_send.shape[1],dim=1))

def _set_boradcast_expert(layer,pl_send,pl_receive,expert_idx):
    pl_send[layer][expert_idx]=True
    for rank_idx in range(WORLD_SIZE):
        if expert_idx//NUM_EXPERTS!=rank_idx:
            pl_receive[layer][expert_idx*WORLD_SIZE+rank_idx]=True

def policy(runtime,pl_send,pl_receive):
    # test
    for layer in range(runtime.num_layers):
        for expert_idx in range(WORLD_SIZE*NUM_EXPERTS):
            _set_boradcast_expert(layer,pl_send,pl_receive,expert_idx)

    pass

def using_policy(runtime):
    pl_send=torch.zeros(runtime.num_layers,WORLD_SIZE*NUM_EXPERTS, dtype=torch.bool)
    pl_receive=torch.zeros(runtime.num_layers,WORLD_SIZE*NUM_EXPERTS*WORLD_SIZE, dtype=torch.bool)

    if runtime.rank==0:
        policy(runtime,pl_send,pl_receive)
        _check_send_receive_keep(runtime.num_layers,runtime.model_keep_time,pl_send,pl_receive,runtime.global_pl_keep)

    return _update_send_receive_keep_on_all_rank(pl_send,pl_receive,runtime.global_pl_keep)
