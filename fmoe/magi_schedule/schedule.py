r"""
The smart schedule proposed in FasterMoE.
"""
import torch
import torch.distributed as dist
import time

from torch.autograd.function import Function

from fmoe.functions import prepare_forward, ensure_comm
from fmoe.functions import _local_scatter, _local_gather 
import fmoe_cuda as fmoe_native
# from fmoe.fastermoe import expert_utils
from magi import expert_utils

class MoEForward(Function):
    @staticmethod
    def forward(
            ctx,
            expert_fn,
            experts,
            inp,
            pos_s, pos_g,
            local_expert_count, global_expert_count,redirect_expert_count,
            send_models,receive_models,keep_models,
            re_send,re_receive,re_unreceive,
            fwd_batch_size, out_batch_size, redirect_batch_size,
            num_experts,
            world_size,
            magi_runtime):
        local_input_buf = _local_scatter(inp, pos_s)

        magi_expert=magi_runtime.magi_expert
        ctx.magi_runtime=magi_runtime
        # ctx gibs gobs : 
        # num_experts * world_size:original experts
        # num_experts * world_size:receive experts
        # num_experts * world_size:keep experts
        # num_experts * world_size * world_size:redirect
        ctx.gibs = [None] * (num_experts * world_size * 3+num_experts * world_size * world_size)
        ctx.gobs = [None] * (num_experts * world_size * 3+num_experts * world_size * world_size)
        ctx.experts = experts
        # ctx.expert_size = expert_utils.get_expert_param_size(experts[0])
        ctx.expert_size = magi_expert.expert_size

        # for i in range(num_experts):
        #     assert ctx.expert_size == expert_utils.get_expert_param_size(experts[i]), "report expert_size bug"            

        def _expert_forward(x, y, expert_idx, store_idx ,magi_flag):
            nothing = lambda a: a
            x = x.data

            # get real stored expert_idx from magi_expert
            if magi_flag:
                expert_idx=magi_expert.get_magi_expert_idx(expert_idx)
                
            with torch.enable_grad():
                x.requires_grad = True
                try:
                    # To skip torch autograd's version check.
                    with torch.autograd.graph.saved_tensors_hooks(nothing, nothing):
                        y0 = expert_fn(x, torch.tensor([x.shape[0]], dtype=torch.int64), expert_idx)
                except Exception as e:
                    # Ignore the error and fall back for compatibility to older
                    # versions of PyTorch
                    y0 = expert_fn(x, torch.tensor([x.shape[0]], dtype=torch.int64), expert_idx)
            ctx.gibs[store_idx] = x
            ctx.gobs[store_idx] = y0
            y.copy_(y0)

        # replace by registe_magi_expert_fn push_magi_expert_fn
        # get_param_fn = lambda out, idx: expert_utils.get_expert_params(experts, out, idx)

        # pop_fn = lambda global_expert_idx: expert_utils.pop_expert_params(experts[global_expert_idx])

        # replace by push_magi_expert_fn
        # def stash_fn(params, store_idx, expert_idx):
        #     expert_utils.stash_expert_params(experts, params, expert_idx)
        #     ctx.shadows[store_idx] = params

        # def push_magi_expert_fn(buffer,global_expert_idx,receive_or_keep=0):
        #     if receive_or_keep==0:
        #         magi_expert.push_magi_expert(experts,buffer,global_expert_idx)
        #         ctx.receive[stash_expert_idx]=magi_expert.get_magi_expert_params(experts,global_expert_idx,local_input_buf)
        #     else:
        #         ctx.keep[stash_expert_idx]=magi_expert.get_magi_expert_params(experts,global_expert_idx,local_input_buf)

        local_output_buf, gib = fmoe_native.smart_sch_forward(
                local_input_buf,
                local_expert_count, global_expert_count, redirect_expert_count,
                send_models, receive_models, keep_models,
                re_send, re_receive, re_unreceive,
                fwd_batch_size, redirect_batch_size,
                ctx.expert_size, world_size,
                magi_runtime.magi_profile_flag, magi_runtime.magi_redirect,
                _expert_forward, magi_expert.registe_magi_expert, magi_expert.push_magi_expert, magi_runtime.record_fwd_layer_time)
        
        out = _local_gather(local_output_buf, pos_g, out_batch_size,
                maybe_overlap=False)
        
        # gib and local_input_buf are necessary, because ctx.gibs are created
        # based on their memory
        variables = (pos_s, pos_g, local_expert_count, global_expert_count, redirect_expert_count,
                send_models, receive_models, keep_models,re_send,re_receive,re_unreceive, gib, local_input_buf)
        ctx.moe_args = fwd_batch_size, inp.shape[0],redirect_batch_size, num_experts, world_size
        ctx.save_for_backward(*variables)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        ctx.magi_runtime.pre_layer()
        (pos_s, pos_g, local_expert_count, global_expert_count, redirect_expert_count,
        send_models, receive_models, keep_models,re_send,re_receive,re_unreceive, _, local_input_buf) = ctx.saved_tensors
        (fwd_batch_size, inp_batch_size, redirect_batch_size, num_experts, world_size) = ctx.moe_args
        magi_expert=ctx.magi_runtime.magi_expert
        experts = ctx.experts
        receive_grads=[None] * world_size * num_experts
        keep_grads=[None] * world_size * num_experts
        def _expert_backward(grad_y, grad_x, expert_idx, store_idx ,magi_flag):
            y = ctx.gobs[store_idx]
            x = ctx.gibs[store_idx]
            # assert grad_y is None or y is None or x is None, "backward parameter bug"
            torch.autograd.backward([y], [grad_y])
            grad_x.copy_(x.grad)
        
        # def stash_fn(store_idx, expert_idx):
        #     expert_utils.stash_expert_params(experts, ctx.shadows[store_idx], expert_idx)

        # def is_magi_expert_exist_fn(flag_buf,rank_idx,expert_idx):
        #     ctx.magi_runtime.is_magi_expert_exist(flag_buf,rank_idx,expert_idx,layer=layer)
        
        # def is_global_magi_expert_exist_fn(flag_buf,expert_idx):
        #     ctx.magi_runtime.is_global_magi_expert_exist(flag_buf,expert_idx,layer=layer)
        
        # pop_fn = lambda idx: expert_utils.pop_expert_params(experts, idx)
        # pop_fn = lambda global_expert_idx: expert_utils.pop_expert_params(experts[global_expert_idx])

        def collect_fn(global_expert_idx,magi_flag,receive_or_keep):
            return
            if receive_or_keep==0:
                receive_grads[global_expert_idx]=local_input_buf.new_zeros(ctx.expert_size)
                if magi_flag:
                    expert_idx = magi_expert.get_magi_expert_idx(global_expert_idx)
                    expert_utils.collect_expert_grads(experts[expert_idx], receive_grads[global_expert_idx])
                fmoe_native.reduce_grad(receive_grads[global_expert_idx], ctx.expert_size)

            elif receive_or_keep==1:
                keep_grads[global_expert_idx]=local_input_buf.new_zeros(ctx.expert_size)
                if magi_flag:
                    expert_idx = magi_expert.get_magi_expert_idx(global_expert_idx)
                    expert_utils.collect_expert_grads(experts[expert_idx], keep_grads[global_expert_idx])
                fmoe_native.reduce_grad(keep_grads[global_expert_idx], ctx.expert_size)


        def set_grad_fn(global_expert_idx,receive_or_keep_or_send):
            return
            if receive_or_keep_or_send==0:
                expert_utils.set_grads(experts[magi_expert.get_magi_expert_idx(global_expert_idx)], receive_grads[global_expert_idx]) 
            elif receive_or_keep_or_send==1:
                expert_utils.set_grads(experts[magi_expert.get_magi_expert_idx(global_expert_idx)], keep_grads[global_expert_idx]) 
            elif receive_or_keep_or_send==2:
                expert_utils.set_grads(experts[global_expert_idx%num_experts], receive_grads[global_expert_idx]) 

        grad_out_buf = _local_scatter(grad_out.contiguous(), pos_g)
        grad_in_buf = fmoe_native.smart_sch_backward(
                grad_out_buf,
                local_expert_count, global_expert_count,redirect_expert_count,
                send_models, receive_models,keep_models,
                re_send,re_receive, re_unreceive,
                pos_s.shape[0], fwd_batch_size, redirect_batch_size,
                world_size,
                ctx.magi_runtime.magi_profile_flag, ctx.magi_runtime.magi_redirect,
                _expert_backward, collect_fn, set_grad_fn, ctx.magi_runtime.record_bwd_layer_time)
        grad_in = _local_gather(grad_in_buf, pos_s, inp_batch_size)

        return (None, None, grad_in, None, None, None,None, None, None, None, None, None, None, None, None, None, None, None, None, None)

policy_fn = None

def _fmoe_general_global_forward(inp, gate, expert_fn, n_expert, world_size, experts=None, magi_runtime=None):
    # TODO: Using multiple tensors as input is to be supported.
    assert(isinstance(inp, torch.Tensor))
    (
        pos,
        local_expert_count,
        global_expert_count,
        fwd_expert_count,
        fwd_batch_size,
    ) = prepare_forward(gate, n_expert, world_size)

    # global policy_fn
    # if policy_fn is None:
    #     policy_fn = get_magi_policy()

    # stored_models is replace by send_models receive_models
    # if stored_models is None:
    #     stored_models = policy_fn(local_expert_count, global_expert_count, # the res of policy_fn
    #             n_expert, world_size, inp.device)

    # magi params
    send_models=magi_runtime.get_send_models()
    receive_models=magi_runtime.get_receive_models()
    keep_models=magi_runtime.get_keep_models()
    
    # token redirect params
    re_send,re_receive,re_unreceive=magi_runtime.get_redirect_models()
    redirect_expert_count=magi_runtime.get_redirect_expert_count(re_receive)
    redirect_batch_size=int(redirect_expert_count.sum().item())

    topk = 1
    if len(gate.shape) == 2:
        topk = gate.shape[1]
    out_batch_size = inp.shape[0] * topk

    return MoEForward.apply(expert_fn, experts, inp,
            torch.div(pos, topk, rounding_mode='floor'), pos,
            local_expert_count, global_expert_count,redirect_expert_count,
            send_models, receive_models,keep_models,
            re_send,re_receive,re_unreceive,
            fwd_batch_size, out_batch_size, redirect_batch_size, n_expert, world_size,magi_runtime)
