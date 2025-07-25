import torch
import os
import numpy as np
import time

PRINT_ALL_RANK=0
PRINT_TIME=1
PRINT_SEND_DEL=0
PRINT_POLICY_TENSOR=0

SAVE_ALL_GLOBAL_TOKEN_LOG=1
SAVE_STD_LOG=1 # origin_token and receive_token also save in this log
SAVE_KEEP_LOG=1

PRINT_RANK=None
RANK=None
MAGI_PROFILER=0
POLICY=None
FILE_NAME_TAIL=None

def init_log(runtime):
    global PRINT_RANK
    global RANK
    global MAGI_PROFILER
    global POLICY
    global FILE_NAME_TAIL
    PRINT_RANK=runtime.world_size-1
    RANK=runtime.rank
    MAGI_PROFILER=runtime.magi_profile_flag
    torch.set_printoptions(linewidth=99999999,threshold=99999999)
    if runtime.janus:
        POLICY="JANUS"
    elif runtime.fastermoe:
        POLICY="SFASTER"
    else:
        POLICY=f"MAGI-{runtime.magi_policy}-{runtime.policy_interval}-{runtime.proxy_expert_ratio if runtime.proxy_expert_ratio!=1.0 else '1'}"
    FILE_NAME_TAIL=f"gate-{runtime.gate}_ws-{runtime.world_size}_layer-{runtime.num_layers}_bs-{runtime.global_batch_size}_topk-{runtime.topk}_sq-{runtime.seq_length}_ep-{runtime.num_experts}_hidden-{runtime.d_model}.log"

    if runtime.rank==runtime.world_size-1 and os.path.exists(f"logs/{runtime.model}/{runtime.model}_{POLICY}_GLOBAL_{FILE_NAME_TAIL}"):
        os.remove(f"logs/{runtime.model}/{runtime.model}_{POLICY}_GLOBAL_{FILE_NAME_TAIL}")

    if runtime.rank==runtime.world_size-1 and os.path.exists(f"logs/{runtime.model}/{runtime.model}_{POLICY}_STD_{FILE_NAME_TAIL}"):
        os.remove(f"logs/{runtime.model}/{runtime.model}_{POLICY}_STD_{FILE_NAME_TAIL}")
    
    if runtime.rank==runtime.world_size-1 and os.path.exists(f"logs/{runtime.model}/{runtime.model}_{POLICY}_KEEP_{FILE_NAME_TAIL}"):
        os.remove(f"logs/{runtime.model}/{runtime.model}_{POLICY}_KEEP_{FILE_NAME_TAIL}")


def print_time(per_itr_record_time,fowd=True,layer=-1):
    if MAGI_PROFILER:
        if PRINT_TIME:
            stime=sum(per_itr_record_time['stime']) if layer==-1 else per_itr_record_time['stime'][layer]
            ctime=sum(per_itr_record_time['ctime']) if layer==-1 else per_itr_record_time['ctime'][layer]
            ctime_wait=sum(per_itr_record_time['ctime_wait']) if layer==-1 else per_itr_record_time['ctime_wait'][layer]
            rtime=sum(per_itr_record_time['rtime']) if layer==-1 else per_itr_record_time['rtime'][layer]
            rtime_wait=sum(per_itr_record_time['rtime_wait']) if layer==-1 else per_itr_record_time['rtime_wait'][layer]
            magi_ctime=sum(per_itr_record_time['magi_ctime']) if layer==-1 else per_itr_record_time['magi_ctime'][layer]
            keep_ctime=sum(per_itr_record_time['keep_ctime']) if layer==-1 else per_itr_record_time['keep_ctime'][layer]

            time_log_str=f" rank:{RANK}"
            time_log_str+=f" layer all" if layer==-1 else f" layer:{layer}"
            time_log_str+=f" s:{stime:6.2f} cw:{ctime_wait:6.2f} c:{ctime:6.2f} r:{rtime:6.2f}"

            if fowd:
                log_str="forward "
                magi_stime=sum(per_itr_record_time['magi_stime']) if layer==-1 else per_itr_record_time['magi_stime'][layer]
                magi_ctime_wait=sum(per_itr_record_time['magi_ctime_wait']) if layer==-1 else per_itr_record_time['magi_ctime_wait'][layer]
                
                magi_log_str=f" total:{max(stime,magi_stime)+max(ctime,magi_ctime+keep_ctime)+rtime:6.2f}"
                magi_log_str+=f" magi_s: {magi_stime:6.2f}"if magi_stime>10 else ""
                magi_log_str+=f" magi_cw:{magi_ctime_wait:6.2f}"if magi_ctime_wait>1 else ""
                magi_log_str+=f" magi_c: {magi_ctime:6.2f}"if magi_ctime>10 else ""
                magi_log_str+=f" keep_c: {keep_ctime:6.2f}"if keep_ctime>10 else ""
            else:
                log_str="backward"
                magi_reduce=sum(per_itr_record_time['magi_reduce']) if layer==-1 else per_itr_record_time['magi_reduce'][layer]
                keep_reduce=sum(per_itr_record_time['keep_reduce']) if layer==-1 else per_itr_record_time['keep_reduce'][layer]
                set_gradients=sum(per_itr_record_time['set_gradients']) if layer==-1 else per_itr_record_time['set_gradients'][layer]
                
                magi_log_str=f" total:{stime+max(ctime,magi_ctime+keep_ctime)+max(rtime,magi_reduce+keep_reduce+set_gradients):6.2f}"
                magi_log_str+=f" magi_c: {magi_ctime:6.2f}"if magi_ctime>10 else ""
                magi_log_str+=f" magi_rd:{magi_reduce:6.2f}"if magi_reduce>10 else ""
                magi_log_str+=f" keep_c: {keep_ctime:6.2f}" if keep_ctime>10 else ""
                magi_log_str+=f" keep_rd:{keep_reduce:6.2f}"if keep_reduce>10 else ""
                magi_log_str+=f" set_gradients:{set_gradients:6.2f}"if set_gradients>10 else ""

            _print(log_str+time_log_str+magi_log_str)


            # _print(f"rank:{RANK} layer:{layer} stime:{stime:6.2f} ctime_wait:{ctime_wait:6.2f} ctime:{ctime:6.2f} rtime:{rtime:6.2f} magi_stime:{magi_stime:6.2f} magi_ctime_wait:{magi_ctime_wait:6.2f} magi_ctime:{magi_ctime:6.2f} keep_ctime:{keep_ctime:6.2f} total_time:{stime+ctime+rtime+magi_stime+magi_ctime+keep_ctime:6.2f}")

def save_global_token_log(runtime,all_global_expert_count):
    if SAVE_ALL_GLOBAL_TOKEN_LOG and runtime.rank==runtime.world_size-1:
        with open(f"logs/{runtime.model}/{runtime.model}_{POLICY}_GLOBAL_{FILE_NAME_TAIL}",'a') as f:
            f.write(f"itr:{runtime.itr} layer:{runtime.layer} all_global_expert_count:{all_global_expert_count}\n")

def save_or_token_log(runtime,receive_token,origin_token): 
    if SAVE_STD_LOG and runtime.rank==runtime.world_size-1:
        with open(f"logs/{runtime.model}/{runtime.model}_{POLICY}_STD_{FILE_NAME_TAIL}",'a') as f:
            f.write(f"itr:{runtime.itr} layer:{runtime.layer} total_input:{runtime.total_input_size} recv_avg:{np.average(receive_token):.2f} recv_std:{np.std(receive_token, ddof=1):.2f} recv_token:{receive_token} origin_token:{origin_token}\n")

def save_keep_log(runtime, global_pl_keep):
    if SAVE_KEEP_LOG and runtime.rank==runtime.world_size-1:
        with open(f"logs/{runtime.model}/{runtime.model}_{POLICY}_KEEP_{FILE_NAME_TAIL}",'a') as f:
            global_keep=torch.cat(global_pl_keep,dim=1)
            for layer in range(runtime.num_layers):
                f.write(f"itr:{runtime.itr} layer:{layer} keep:{global_keep[layer]}\n")

def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start
        print(f"rank:{RANK} func:{func.__name__} use {duration*1000:.2f}ms")
        return result
    return wrapper

def print_policy_tensor(msg):
    if PRINT_POLICY_TENSOR:
        _print(msg)

def send_del_log(msg):
    if PRINT_SEND_DEL:
        _print(msg)

def _print(msg):
    if PRINT_ALL_RANK:
        print(msg)
    elif PRINT_RANK==RANK:
        print(f"{msg}")