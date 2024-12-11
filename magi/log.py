PRINT_TIME=1
PRINT_TOKEN=0
PRINT_RANK_0=0
SAVE_GLOBAL_TOKEN_LOG=1
RANK=None

def set_rank(rank):
    global RANK
    RANK=rank

def print_time(per_itr_record_time,layer=0):
    if PRINT_TIME:
        if layer==-1:
            stime=sum(per_itr_record_time['stime'])
            ctime=sum(per_itr_record_time['ctime'])
            rtime=sum(per_itr_record_time['rtime'])
            shadow_stime=sum(per_itr_record_time['shadow_stime'])
            shadow_ctime=sum(per_itr_record_time['shadow_ctime'])
        else:
            stime=per_itr_record_time['stime'][layer]
            ctime=per_itr_record_time['ctime'][layer]
            rtime=per_itr_record_time['rtime'][layer]
            shadow_stime=per_itr_record_time['shadow_stime'][layer]
            shadow_ctime=per_itr_record_time['shadow_ctime'][layer]

        _print(f"rank:{RANK} layer:{layer} stime:{stime:.2f} ctime:{ctime:.2f} rtime:{rtime:.2f} shadow_stime:{shadow_stime:.2f} shadow_ctime:{shadow_ctime:.2f}")

def save_global_token_log(gate,layer,itr,global_expert_count):
    if SAVE_GLOBAL_TOKEN_LOG:
        with open(f"log/{gate}_{RANK}_global_token_count",'a') as f:
            f.write(f"layer:{layer} itr:{itr} global_expert_count:{global_expert_count}\n")

def print_token(itr,layer,recive_token,origin_token): 
    if PRINT_TOKEN:        
        _print(f"itr :{itr} layer:{layer} recive_token:{recive_token} origin_token:{origin_token}")

def _print(msg):
    if PRINT_RANK_0:
        if RANK==0:
            print(f"\n{msg}\n")
    else:
        print(msg)