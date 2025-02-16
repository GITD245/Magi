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

def print_time(per_itr_record_time,layer=-1):
    if MAGI_PROFILER:
        if PRINT_TIME:
            if layer==-1:
                stime=sum(per_itr_record_time['stime'])
                ctime=sum(per_itr_record_time['ctime'])
                ctime_wait=sum(per_itr_record_time['ctime_wait'])
                rtime=sum(per_itr_record_time['rtime'])
                rtime_wait=sum(per_itr_record_time['rtime_wait'])
                magi_stime=sum(per_itr_record_time['magi_stime'])
                magi_ctime=sum(per_itr_record_time['magi_ctime'])
                magi_ctime_wait=sum(per_itr_record_time['magi_ctime_wait'])
                keep_ctime=sum(per_itr_record_time['keep_ctime'])
            else:
                stime=per_itr_record_time['stime'][layer]
                ctime=per_itr_record_time['ctime'][layer]
                ctime_wait=per_itr_record_time['ctime_wait'][layer]
                rtime=per_itr_record_time['rtime'][layer]
                rtime_wait=per_itr_record_time['rtime_wait'][layer]
                magi_stime=per_itr_record_time['magi_stime'][layer]
                magi_ctime=per_itr_record_time['magi_ctime'][layer]
                magi_ctime_wait=per_itr_record_time['magi_ctime_wait'][layer]
                keep_ctime=per_itr_record_time['keep_ctime'][layer]

            _print(f"rank:{RANK} layer:{layer} stime:{stime:.2f} ctime_wait:{ctime_wait:.2f} ctime:{ctime:.2f} rtime:{rtime:.2f} magi_stime:{magi_stime:.2f} magi_ctime_wait:{magi_ctime_wait:.2f} magi_ctime:{magi_ctime:.2f} keep_ctime:{keep_ctime:.2f} total_time:{stime+ctime+rtime+magi_stime+magi_ctime+keep_ctime:.2f}")

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

def profile(msg):
    if MAGI_PROFILER:
        _print(msg)

def send_del_log(msg):
    if PRINT_SEND_DEL:
        _print(msg)

def _print(msg):
    if PRINT_RANK==-1:
        print(msg)
    elif PRINT_RANK==RANK:
        print(f"\n{msg}")