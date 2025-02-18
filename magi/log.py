PRINT_RANK=-1

PRINT_TIME=1
PRINT_TOKEN=0
PRINT_SEND_DEL=0
PRINT_POLICY_TENSOR=0

SAVE_GLOBAL_TOKEN_LOG=0

RANK=None
MAGI_PROFILER=0

def init_log(rank,magi_profile_flag):
    global RANK
    global MAGI_PROFILER
    RANK=rank
    MAGI_PROFILER=magi_profile_flag

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

def save_global_token_log(gate,layer,itr,global_expert_count):
    if SAVE_GLOBAL_TOKEN_LOG:
        with open(f"log/{gate}_{RANK}_global_token_count",'a') as f:
            f.write(f"layer:{layer} itr:{itr} global_expert_count:{global_expert_count}\n")

def print_token(itr,layer,recive_token,origin_token): 
    if PRINT_TOKEN:        
        _print(f"itr :{itr} layer:{layer} recive_token:{recive_token} origin_token:{origin_token}")

def print_policy_tensor(msg):
    if PRINT_POLICY_TENSOR:
        _print(msg)

def send_del_log(msg):
    if PRINT_SEND_DEL:
        _print(msg)

def _print(msg):
    if PRINT_RANK==-1:
        print(msg)
    elif PRINT_RANK==RANK:
        print(f"{msg}")