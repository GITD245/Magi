import torch
import os
import numpy as np

PRINT_ALL_RANK=0
PRINT_TIME=1
PRINT_SEND_DEL=0
PRINT_POLICY_TENSOR=0

SAVE_ALL_GLOBAL_TOKEN_LOG=1
SAVE_OR_TOKEN=1

PRINT_RANK=None
RANK=None
MAGI_PROFILER=0
FILE_NAME_TAIL=None

def init_log(runtime):
    global PRINT_RANK
    global RANK
    global MAGI_PROFILER
    global FILE_NAME_TAIL
    PRINT_RANK=runtime.world_size-1
    RANK=runtime.rank
    MAGI_PROFILER=runtime.magi_profile_flag
    torch.set_printoptions(linewidth=10000000)

    FILE_NAME_TAIL=f"{runtime.model}_{runtime.model}_gate-{runtime.gate}_ws-{runtime.world_size}_layer-{runtime.num_layers}_bs-{runtime.global_batch_size}_topk-{runtime.topk}_sq-{runtime.seq_length}_ep-{runtime.num_experts}_hidden-{runtime.d_model}.log"

    if runtime.rank==runtime.world_size-1 and os.path.exists(f"logs/{runtime.model}/token_count_{FILE_NAME_TAIL}"):
            os.remove(f"logs/{runtime.model}/token_count_{FILE_NAME_TAIL}")

    if runtime.rank==runtime.world_size-1 and os.path.exists(f"logs/{runtime.model}/or_token_count_{FILE_NAME_TAIL}"):
            os.remove(f"logs/{runtime.model}/or_token_count_{FILE_NAME_TAIL}")


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
        with open(f"logs/{runtime.model}/token_count_{FILE_NAME_TAIL}",'a') as f:
            f.write(f"itr:{runtime.itr} layer:{runtime.layer} all_global_expert_count:{all_global_expert_count}\n")

def save_or_token(runtime,receive_token,origin_token): 
    if SAVE_OR_TOKEN and runtime.rank==runtime.world_size-1:
        with open(f"logs/{runtime.model}/or_token_count_{FILE_NAME_TAIL}",'a') as f:
            avg=sum(receive_token)/len(receive_token)
            avg_recieve_token=[x-avg for x in receive_token]
            f.write(f"itr:{runtime.itr} layer:{runtime.layer} recv_rate:{sum(receive_token)/runtime.total_input_size:.2f} var:{np.var(receive_token)} avg_var:{np.var(avg_recieve_token)} recv_token:{receive_token}\n")

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