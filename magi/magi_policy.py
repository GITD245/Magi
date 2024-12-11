import torch



def policy(profile):

    send_models=torch.zeros(profile.get_world_size()*profile.get_num_experts(), dtype=torch.bool)
    keep_models=torch.zeros(profile.get_world_size()*profile.get_num_experts(), dtype=torch.bool)
    del_models=torch.zeros(profile.get_world_size()*profile.get_num_experts(), dtype=torch.bool)
    send_map={}

    # test
    # send_models[7]=1
    # send_models[1]=1
    # send_models[0]=1

    return send_models,keep_models,del_models,send_map

def get_magi_policy():
    return policy