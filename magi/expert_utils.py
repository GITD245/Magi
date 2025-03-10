import torch

def get_expert_param_size(e):
    return sum(map(lambda x: x.numel(), e.parameters()))

def get_params(expert,out):
    offset = 0
    for n, p in expert.named_parameters():
        seg = out[offset:offset + p.numel()]
        offset += p.numel()
        seg.copy_(p.data.flatten())

# def stash_expert_params(e, params, idx):
#     e = e[idx]
#     if not hasattr(e, 'expert_param_stash'):
#         setattr(e, 'expert_param_stash', dict())
#         setattr(e, 'expert_grad_stash', dict())
#     offset = 0
#     for n, p in e.named_parameters():
#         if n not in e.expert_param_stash:
#             e.expert_param_stash[n] = p.data.clone()
#             e.expert_grad_stash[n] = p.grad.clone() if p.grad is not None else None
#         with torch.no_grad():
#             seg = params[offset:offset + p.numel()]
#             offset += p.numel()
#             p.copy_(seg.reshape(p.shape))
#             p.grad = None

# def pop_expert_params(expert):
#         if not hasattr(expert, 'expert_param_stash'):
#             return
#         if not expert.expert_param_stash:
#             return
#         for n, p in expert.named_parameters():
#             with torch.no_grad():
#                 p.copy_(expert.expert_param_stash[n])
#                 if expert.expert_grad_stash[n] is not None:
#                     p.grad = expert.expert_grad_stash[n].clone()
#         expert.expert_param_stash.clear()
#         expert.expert_grad_stash.clear()

def push_params(params,expert):
    offset = 0
    for n, p in expert.named_parameters():
        with torch.no_grad():
            seg = params[offset:offset + p.numel()]
            offset += p.numel()
            p.copy_(seg.reshape(p.shape))
            p.grad = None

def collect_expert_grads(e, grads):
    offset = 0
    for _, p in e.named_parameters():
        seg = grads[offset:offset + p.numel()]
        offset += p.numel()
        if p.grad is not None:
            seg.copy_(p.grad.flatten())
            p.grad = None
        else:
            seg.zero_()

def set_grads(e, grads):
    offset = 0
    for n, p in e.named_parameters():
        seg = grads[offset:offset + p.numel()]
        offset += p.numel()
        if p.grad is None:
            p.grad = seg.clone().reshape(p.shape)
        else:
            p.grad += seg.reshape(p.shape)
