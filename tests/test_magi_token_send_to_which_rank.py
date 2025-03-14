# import torch
world_size=8
num_experts=2
global_pl_keep=list()


def gen_global_pl_keep():
    global global_pl_keep
    global_pl_keep=[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,
                    1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0,
                    1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,
                    0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0]

def get_global_keep_models_nums(expert_idx):
    # origin expert rank is not cnt in keep_models
    cnt=1
    for rank_idx in range(world_size):
        if global_pl_keep[rank_idx*world_size*num_experts+expert_idx]>0:
            cnt+=1
    return cnt


def get_redirect_models():
    cnt_send,cnt_receive, cnt_unreceive=0,0,0
    for rank in range(world_size):
        re_send=[-1]*(num_experts*world_size)
        re_receive=[0 for _ in range(num_experts*world_size*world_size)]
        re_unreceive=[0 for _ in range(num_experts*world_size*world_size)]

        # re_send: change expert_idx's token target from global origin expert to global magi expert
        # re_receive: expert_idx's token receive for local magi expert
        # re_unreceive: expert_idx's token unreceive for local origin expert

        local_keep_models=global_pl_keep[rank*world_size*num_experts:(1+rank)*world_size*num_experts]
        for expert_idx in range(world_size*num_experts):
            origin_rank=expert_idx//num_experts
            global_keep_models_nums=get_global_keep_models_nums(expert_idx)
            keep_rank_interval=world_size//global_keep_models_nums

            # re_send
            if global_keep_models_nums>1:
                # expert_idx exist global magi models
                send_rank=(origin_rank+keep_rank_interval*\
                                        (rank//keep_rank_interval-origin_rank//keep_rank_interval))\
                                        %world_size
                # don't send local token to local magi/origin expert
                # don't send local token to global origin expert
                if (send_rank!=rank and send_rank!=origin_rank):
                    re_send[expert_idx]=send_rank

            # re_receive
            if local_keep_models[expert_idx]>0: 
                # local has expert_idx's magi expert
                keep_rank=rank//keep_rank_interval*keep_rank_interval
                re_receive[expert_idx*world_size+keep_rank:expert_idx*world_size+keep_rank+keep_rank_interval]=[1]*keep_rank_interval
                # don't receive local token for local magi/origin expert
                re_receive[expert_idx*world_size+rank]=0
                # don/t receive global token for local origin expert
                re_receive[expert_idx*world_size+origin_rank]=0
                
            
            # re_unreceive
            if global_keep_models_nums>1 and origin_rank==rank:
                # expert_idx exist global magi models and local has origin expert
                keep_rank=rank//keep_rank_interval*keep_rank_interval
                re_unreceive[expert_idx*world_size:expert_idx*world_size+world_size]=[1]*world_size
                re_unreceive[expert_idx*world_size+keep_rank:expert_idx*world_size+keep_rank+keep_rank_interval]=[0]*keep_rank_interval
                # receive global token for local origin expert
                offset=rank%keep_rank_interval
                for i in range(expert_idx*world_size+offset,expert_idx*world_size+offset+world_size,keep_rank_interval):
                    re_unreceive[i]=0

            

        # res[expert_idx]=rank_idx means this rank's expert_idx's token should send to rank_idx
        print(f'rank:{rank}')
        print(f're_send:{re_send}')
        for i in range(world_size*num_experts):
            if re_send[i]!=-1:
                cnt_send+=1

        print(f're_receive:')
        for i in range(world_size*num_experts):
            print(f'expert_{i}: ',end=' ')
            for j in range(world_size):
                print(re_receive[i*world_size+j],end=' ')
                if re_receive[i*world_size+j]!=0:
                    cnt_receive+=1
            print()

        print(f're_unreceive:')
        for i in range(world_size*num_experts):
            print(f'expert_{i}: ',end=' ')
            for j in range(world_size):
                print(re_unreceive[i*world_size+j],end=' ')
                if re_unreceive[i*world_size+j]!=0:
                    cnt_unreceive+=1
            print()
        print()
    print(f'cnt_send:{cnt_send}, cnt_receive:{cnt_receive}, cnt_unreceive:{cnt_unreceive},three numbers should be equal')
gen_global_pl_keep()
get_redirect_models()
# get_redirect_token_receive_from_which_rank()
# re_send 取消发送/新增放松
# re_unreceive
# re_receive 新增接收
# 取消接收还没实现