#ifdef FMOE_USE_NCCL

#include <c10/cuda/CUDAGuard.h>
#include <cstdlib>
#include <torch/extension.h>
#include <vector>

#include "smart_schedule.h"
#include "status.h"

long pipeline_gran = -1;

int smart_sch_enabled = 0;

int isSmartSchEnabled() { return smart_sch_enabled; }
void setSmartSchEnabled(int s) { smart_sch_enabled = s; }

inline ncclDataType_t getNcclDataType(at::ScalarType t) {
  switch (t) {
  case at::kChar:
    return ncclInt8;
  case at::kByte:
    return ncclUint8;
  case at::kFloat:
    return ncclFloat;
  case at::kDouble:
    return ncclDouble;
  case at::kInt:
    return ncclInt32;
  case at::kLong:
    return ncclInt64;
  case at::kHalf:
    return ncclHalf;
  case at::kBool:
    return ncclUint8;
#if defined(ENABLE_NCCL_BF16_DATATYPE)
  case at::kBFloat16:
    return ncclBfloat16;
#endif
  default:
    return ncclChar;
  }
}

void _reduce_grad(torch::Tensor t, long expert_size) {
  auto smgr = getCudaStreamManager(t.device().index());

  cudaEvent_t evt_stash;
  cudaEventCreate(&evt_stash);
  cudaEventRecord(evt_stash, smgr->torchStream());
  FMOE_SWE(smgr->stream(0), evt_stash);
  cudaEventDestroy(evt_stash);

  auto dtype = getNcclDataType(t.scalar_type());
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, t.scalar_type(),
      "fmoe_cuda_reduce_grad", ([&] {
        void *buf = (void *)t.data_ptr<scalar_t>();
        NCCL_SAFE_CALL(ncclAllReduce(buf, buf, expert_size, dtype, ncclSum,
                                     smgr->ncclcomm, smgr->stream(0)));
      }));
}

std::vector<torch::Tensor> _smart_sch_forward(
    torch::Tensor input_buf, // fwd_batch_size*1024
    torch::Tensor local_expert_count, torch::Tensor global_expert_count, torch::Tensor redirect_expert_count,
    torch::Tensor send_models, torch::Tensor receive_models, torch::Tensor keep_models,
    torch::Tensor re_send, torch::Tensor re_receive, torch::Tensor re_unreceive,
    long global_batch_size, long redirect_batch_size,
    long expert_size, long n_workers,
    bool magi_profile_flag, bool magi_redirect_flag,
    py::function forward_fn, py::function registe_magi_expert_fn, py::function push_magi_expert_fn, py::function record_layer_time_fn) {
  if (pipeline_gran == -1) {
    char *p = getenv("FMOE_FASTER_GROUP_SIZE");
    if (p) {
      pipeline_gran = atoi(p);
    } else {
      pipeline_gran = 4;
    }
    setSmartSchEnabled(1);
  }

  auto smgr = getCudaStreamManager(input_buf.device().index());
  int rank;
  NCCL_SAFE_CALL(ncclCommUserRank(smgr->ncclcomm, &rank));

  const auto num_experts = local_expert_count.size(0) / n_workers;
  const auto d_model = input_buf.size(1);
  const long redirect_expert_nums = redirect_expert_count.size(0);

  // TODO: maybe empty is faster
  auto global_input_buf = input_buf.new_zeros({global_batch_size, d_model});
  auto redirect_input_buf = input_buf.new_zeros({redirect_batch_size, d_model});

  auto output_buf = input_buf.new_zeros({input_buf.size(0), d_model});
  auto global_output_buf = input_buf.new_zeros({global_batch_size, d_model});
  auto redirect_output_buf = input_buf.new_zeros({redirect_batch_size, d_model});

  std::vector<torch::Tensor> send_params;
  std::vector<torch::Tensor> receive_params;
  auto send_models_ = send_models.data_ptr<bool>();
  auto receive_models_ = receive_models.data_ptr<bool>();
  for (long i = 0; i < num_experts * n_workers;
       ++i) { // get shadow_expert to params
    if (send_models_[i]) {
      torch::Tensor t = input_buf.new_empty({expert_size});
      if (i / num_experts == rank) {
        // send expert param buffer
        registe_magi_expert_fn(t, i, 1);
        send_params.push_back(t);
      } else {
        if (receive_models_[i * n_workers + rank]) {
          // receive expert param buffer
          registe_magi_expert_fn(t, i, 0);
          receive_params.push_back(t);
        }
      }
    }
  }

  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16, input_buf.scalar_type(),
      "fmoe_cuda_smart_sch_forward", ([&] {
        fmoe_cuda_fused_forward_impl(
            forward_fn, record_layer_time_fn, push_magi_expert_fn,
            input_buf.device(),

            send_params, receive_params,

            input_buf.data_ptr<scalar_t>(),
            global_input_buf.data_ptr<scalar_t>(),
            redirect_input_buf.data_ptr<scalar_t>(),
            output_buf.data_ptr<scalar_t>(),
            global_output_buf.data_ptr<scalar_t>(),
            redirect_output_buf.data_ptr<scalar_t>(),

            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            redirect_expert_count.data_ptr<long>(),

            send_models.data_ptr<bool>(), receive_models.data_ptr<bool>(), keep_models.data_ptr<int>(),

            re_send.data_ptr<int>(), re_receive.data_ptr<bool>(), re_unreceive.data_ptr<bool>(),

            redirect_expert_nums, d_model, num_experts, rank, n_workers, expert_size, pipeline_gran,
            magi_profile_flag, magi_redirect_flag, smgr);
      }));
  return {output_buf, global_input_buf};
}

torch::Tensor
_smart_sch_backward(
    torch::Tensor grad_out,
    torch::Tensor local_expert_count, torch::Tensor global_expert_count, torch::Tensor redirect_expert_count,
    torch::Tensor send_models, torch::Tensor receive_models, torch::Tensor keep_models,
    torch::Tensor re_send, torch::Tensor re_receive, torch::Tensor re_unreceive,
    long buf_batch_size, long global_batch_size, long redirect_batch_size,
    long n_workers,
    bool magi_profile_flag, bool magi_redirect_flag,
    py::function backward_fn, py::function collect_fn, py::function set_grad_fn, py::function record_layer_time_fn) {
  const auto num_experts = local_expert_count.size(0) / n_workers;
  const long redirect_expert_nums = redirect_expert_count.size(0);
  auto smgr = getCudaStreamManager(grad_out.device().index());
  int rank;
  ncclCommUserRank(smgr->ncclcomm, &rank);
  const auto d_model = grad_out.size(1);

  auto global_grad_out = grad_out.new_zeros({global_batch_size, d_model});
  auto redirect_grad_out = grad_out.new_zeros({redirect_batch_size, d_model});

  auto grad_in = grad_out.new_zeros({buf_batch_size, d_model});
  auto global_grad_in = grad_out.new_zeros({global_batch_size, d_model});
  auto redirect_grad_in = grad_out.new_zeros({redirect_batch_size, d_model});

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      grad_out.scalar_type(), "fmoe_cuda_smartsch_backward", ([&] {
        fmoe_cuda_fused_backward_impl(
            backward_fn, record_layer_time_fn, collect_fn, set_grad_fn,
            grad_out.device(),

            grad_out.data_ptr<scalar_t>(),
            global_grad_out.data_ptr<scalar_t>(),
            redirect_grad_out.data_ptr<scalar_t>(),
            grad_in.data_ptr<scalar_t>(),
            global_grad_in.data_ptr<scalar_t>(),
            redirect_grad_in.data_ptr<scalar_t>(),

            local_expert_count.data_ptr<long>(),
            global_expert_count.data_ptr<long>(),
            redirect_expert_count.data_ptr<long>(),

            send_models.data_ptr<bool>(),
            receive_models.data_ptr<bool>(), keep_models.data_ptr<int>(),
            re_send.data_ptr<int>(), re_receive.data_ptr<bool>(), re_unreceive.data_ptr<bool>(),
            redirect_expert_nums, d_model, num_experts, rank, n_workers, pipeline_gran,
            magi_profile_flag, magi_redirect_flag, smgr);
      }));
  return grad_in;
}
#endif
