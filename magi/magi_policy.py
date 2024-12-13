import torch

def policy(runtime):

    send_models=torch.zeros(runtime.get_world_size()*runtime.get_num_experts(), dtype=torch.bool)
    recive_models=torch.zeros(runtime.get_world_size()*runtime.get_num_experts(), dtype=torch.bool)
    del_models=torch.zeros(runtime.get_world_size()*runtime.get_num_experts(), dtype=torch.bool)
    send_map={}
    # test
    # send_models[7]=1
    if runtime.get_layer()==0:
        send_models[0]=1
        send_models[1]=1
        send_models[2]=1
        send_models[3]=1
    

    return send_models,del_models,send_map

def update_policy(runtime):
    
    pass

def get_magi_policy():
    return policy