#ifndef SMART_SCHEDULE_H
#define SMART_SCHEDULE_H

#include <cstdio>
#include <iostream> // IWYU pragma: keep
#include <vector>   // IWYU pragma: keep

#include <cuda.h>
#include <cuda_runtime.h>
#include <nccl.h>

#include "../stream_manager.h"
#include "c10/core/Device.h"
#include <pybind11/pybind11.h>
#include <torch/torch.h>
namespace py = pybind11;

#if defined(CUDA_VERSION) && (CUDA_VERSION < 110010)
#define FMOE_SWE(__s__, __e__) cudaStreamWaitEvent(__s__, __e__, 0)
#else
#define FMOE_SWE(__s__, __e__) cudaStreamWaitEvent(__s__, __e__)
#endif

#define CUDA_CHECK(call)                                                                                      \
  {                                                                                                           \
    cudaError_t err = call;                                                                                   \
    if (err != cudaSuccess) {                                                                                 \
      std::cerr << "CUDA Error in " << __FILE__ << " at line " << __LINE__ << ": " << cudaGetErrorString(err) \
                << std::endl;                                                                                 \
      exit(err);                                                                                              \
    }                                                                                                         \
  }

template <typename scalar_t>
void exchangeWith(const scalar_t *sendbuf, size_t sendcount, int t_send, scalar_t *recvbuf, size_t recvcount,
                  int t_recv, long d_model, cudaStream_t stream, ncclComm_t comm) {
  if (sendcount) {
    ncclSend(sendbuf, sendcount * d_model * sizeof(scalar_t), ncclChar, t_send, comm, stream);
  }
  if (recvcount) {
    ncclRecv(recvbuf, recvcount * d_model * sizeof(scalar_t), ncclChar, t_recv, comm, stream);
  }
}

#define GEN_BASE(_step)                                           \
  long to_base = (group_rank + _step) % n_groups * pipeline_gran; \
  long from_base = (group_rank + n_groups - _step) % n_groups * pipeline_gran;
#define GEN_IDX                                \
  int idx_send = ei + rank_send * num_experts; \
  int idx_recv = ei + rank_recv * num_experts; \
  int gidx_send = ei * world_size + rank_send; \
  int gidx_recv = ei * world_size + rank_recv; \
  int idx_self = ei + rank * num_experts;

// bool is_magi_expert_exist(py::function is_magi_expert_exist_fn, int rank_idx, int expert_idx) { // NOLINT
//   torch::Tensor bool_tensor = torch::zeros({1}, torch::kBool);
//   is_magi_expert_exist_fn(bool_tensor, rank_idx, expert_idx);
//   bool is_magi_exist_flag = bool_tensor.data_ptr<bool>()[0];
//   return is_magi_exist_flag;
// }

// local_ptr:local token,used in S_0 C_shadow R_0   global_ptr:token need to be
// received,used in S_0 C_0 R_0
inline bool is_global_keep_expert_exist(long num_experts, long world_size, const int *keep_models) {
  bool global_exist_flag = false;
  for (long j = 0; j < num_experts * world_size * world_size; ++j) {
    if (keep_models[j] > 0) {
      global_exist_flag = true;
      break;
    }
  }
  return global_exist_flag;
}

void computePtrs(long num_experts, long rank, long world_size, const long *local_expert_count, // NOLINT
                 const long *global_expert_count, const bool *send_models, const bool *receive_models, const int *keep_models, int *local_ptr, int *global_ptr) {
  local_ptr[0] = global_ptr[0] = 0;

  for (int i = 0; i < num_experts * world_size; ++i) {
    local_ptr[i + 1] = local_ptr[i] + local_expert_count[i];

    auto expert_idx = i % num_experts;
    auto worker_idx = i / num_experts;
    auto global_expert_idx = rank * num_experts + expert_idx;
    auto gp_idx = expert_idx * world_size + worker_idx;

    // if local model wasn't become a magi_model, receive global tokens
    if (receive_models[global_expert_idx * world_size + worker_idx] ||
        (keep_models[num_experts * world_size * worker_idx + global_expert_idx] > 0)) {
      // is_magi_expert_exist(is_magi_expert_exist_fn, worker_idx, global_expert_idx)
      global_ptr[gp_idx + 1] = 0;
    } else {
      global_ptr[gp_idx + 1] = global_expert_count[i];
    }
  }
  global_ptr[0] = 0;
  for (int i = 0; i < num_experts * world_size; ++i) {
    global_ptr[i + 1] += global_ptr[i];
  }
}

template <typename scalar_t>
void computeFn(py::function fn, c10::Device device, scalar_t *inp_buf, scalar_t *out_buf, long expert_idx,
               long store_idx, long offset, long micro_batch_size, long d_model, bool magi_flag,
               CudaStreamManager *smgr) {
  if (micro_batch_size == 0) {
    return;
  }
  auto options =
      torch::TensorOptions().dtype(c10::CppTypeToScalarType<scalar_t>::value).device(device).requires_grad(true);
  auto inp = torch::from_blob(inp_buf + offset * d_model, {micro_batch_size, d_model}, options);
  auto oup = torch::from_blob(out_buf + offset * d_model, {micro_batch_size, d_model}, options);
  smgr->use_default = true;
  fn(inp, oup, expert_idx, store_idx, magi_flag);
  smgr->use_default = false;
}

template <typename scalar_t>
void fmoe_cuda_fused_forward_impl(
    py::function forward_fn, py::function record_layer_time_fn,
    py::function push_magi_expert_fn,
    c10::Device device,

    std::vector<torch::Tensor> send_params, std::vector<torch::Tensor> receive_params,

    scalar_t *input_buf, scalar_t *global_input_buf, scalar_t *global_output_buf,
    scalar_t *output_buf,

    const long *local_expert_count, const long *global_expert_count,

    const bool *send_models, const bool *receive_models, const int *keep_models,

    long d_model, long num_experts, long rank, long world_size, long expert_size,
    long pipeline_gran, bool magi_profile_flag, CudaStreamManager *smgr) {
  smgr->syncTorch();

  int *local_ptr = new int[num_experts * world_size + 1];
  int *global_ptr = new int[num_experts * world_size + 1];
  computePtrs(num_experts, rank, world_size, local_expert_count, global_expert_count, send_models, receive_models, keep_models, local_ptr, global_ptr);
  if (pipeline_gran > world_size) {
    pipeline_gran = world_size;
  }
  long n_groups = world_size / pipeline_gran;
  long group_rank = rank / pipeline_gran;

  cudaEvent_t *input_ready = new cudaEvent_t[n_groups];
  cudaEvent_t *output_ready = new cudaEvent_t[n_groups];
  cudaEvent_t *output_torch_ready = new cudaEvent_t[n_groups];

  cudaEvent_t *stime_start = new cudaEvent_t[n_groups];
  cudaEvent_t *ctime_launch = new cudaEvent_t[n_groups];
  cudaEvent_t *ctime_start = new cudaEvent_t[n_groups];
  cudaEvent_t *rtime_launch = new cudaEvent_t[n_groups];
  cudaEvent_t *rtime_start = new cudaEvent_t[n_groups];
  cudaEvent_t *rtime_end = new cudaEvent_t[n_groups];

  for (long i = 0; i < n_groups; ++i) {
    cudaEventCreate(input_ready + i);
    cudaEventCreate(output_ready + i);
    cudaEventCreate(output_torch_ready + i);

    cudaEventCreate(stime_start + i);
    cudaEventCreate(ctime_launch + i);
    cudaEventCreate(ctime_start + i);
    cudaEventCreate(rtime_launch + i);
    cudaEventCreate(rtime_start + i);
    cudaEventCreate(rtime_end + i);
  }

  cudaEvent_t *magi_stime_start = new cudaEvent_t[world_size * num_experts];
  cudaEvent_t *magi_stime_end = new cudaEvent_t[world_size * num_experts];
  cudaEvent_t *magi_ctime_launch = new cudaEvent_t[world_size * num_experts];
  cudaEvent_t *magi_ctime_start = new cudaEvent_t[world_size * num_experts];
  cudaEvent_t *magi_ctime_end = new cudaEvent_t[world_size * num_experts];
  cudaEvent_t *keep_start = new cudaEvent_t[world_size * num_experts];
  cudaEvent_t *keep_end = new cudaEvent_t[world_size * num_experts];

  for (long i = 0; i < world_size * num_experts; ++i) {
    cudaEventCreate(magi_stime_start + i);
    cudaEventCreate(magi_stime_end + i);
    cudaEventCreate(magi_ctime_launch + i);
    cudaEventCreate(magi_ctime_start + i);
    cudaEventCreate(magi_ctime_end + i);
    cudaEventCreate(keep_start + i);
    cudaEventCreate(keep_end + i);
  }

  // S_0 ... S_n
  for (long step = 0; step < n_groups; ++step) {
    if (magi_profile_flag)
      cudaEventRecord(stime_start[step], smgr->stream(num_experts));
    for (long ei = 0; ei < num_experts; ++ei) {
      GEN_BASE(step);
      NCCL_SAFE_CALL(ncclGroupStart());
      for (int j = 0; j < pipeline_gran; ++j) {
        int rank_send = j + to_base;
        int rank_recv = j + from_base;
        GEN_IDX;
        // MAGI_TODO 本地计算的token也会send一趟
        // 也许直接从input_buf复制到global_input_buf性能会更好？ if send
        // worker(send to local) has magi_expert(receive or keep), no need to
        // send
        exchangeWith(input_buf + local_ptr[idx_send] * d_model,
                     local_expert_count[idx_send] * !receive_models[idx_send * world_size + rank] *
                         !(keep_models[num_experts * world_size * rank + idx_send] > 0),
                     //  !is_magi_expert_exist(is_magi_expert_exist_fn,keep_models, rank,
                     //  idx_send),
                     rank_send,

                     // if recv worker(receive from global) has
                     // magi_expert(receive or keep), no need to receive
                     global_input_buf + global_ptr[gidx_recv] * d_model,
                     global_expert_count[idx_recv] * !receive_models[idx_self * world_size + rank_recv] *
                         !(keep_models[num_experts * world_size * rank_recv + idx_self] > 0),
                     // !is_magi_expert_exist(is_magi_expert_exist_fn,rank_recv,idx_self),
                     rank_recv,

                     d_model, smgr->stream(num_experts), smgr->ncclcomm);
      }
      NCCL_SAFE_CALL(ncclGroupEnd());
    }
    cudaEventRecord(input_ready[step], smgr->stream(num_experts));
  }
  // Broadcast shadowed experts
  // cudaEvent_t evt_get, *evt_shadow;
  // if (send_params.size() > 0) {
  //     evt_shadow = new cudaEvent_t[send_params.size()];
  // }
  // for (long i = 0, si = 0; i < world_size * num_experts; ++i) {
  //     if ((stored_models[i] && (i/num_experts)==rank) ||
  //     (receive_models[i*world_size+rank])) {
  //         if (magi_profile_flag) cudaEventRecord(magi_stime_start[i],
  //         smgr->stream(num_experts)); if (i / num_experts == rank) {
  //             cudaEventCreate(&evt_get);
  //             cudaEventRecord(evt_get, smgr->stream(0));
  //             FMOE_SWE(smgr->stream(num_experts), evt_get);
  //             cudaEventDestroy(evt_get);
  //         }
  //         NCCL_SAFE_CALL(ncclBcast((void*)send_params[si].data_ptr<scalar_t>(),
  //                     expert_size * sizeof(scalar_t), ncclChar,
  //                     i / num_experts, smgr->ncclcomm,
  //                     smgr->stream(num_experts)));
  //         cudaEventCreate(evt_shadow + si);
  //         cudaEventRecord(evt_shadow[si], smgr->stream(num_experts));
  //         ++si;
  //         if (magi_profile_flag) cudaEventRecord(magi_stime_end[i],
  //         smgr->stream(num_experts));
  //     }
  // }

  // partial broadcast magi experts
  cudaEvent_t evt_magi_get, *evt_magi_receive;
  if (receive_params.size() > 0) {
    evt_magi_receive = new cudaEvent_t[receive_params.size()];
  }
  for (long i = 0, send_params_idx = 0, receive_params_idx = 0; i < world_size * num_experts; ++i) {
    if (magi_profile_flag)
      cudaEventRecord(magi_stime_start[i], smgr->stream(num_experts));
    if (send_models[i]) {
      NCCL_SAFE_CALL(ncclGroupStart());
      // send part
      if (i / num_experts == rank) {
        cudaEventCreate(&evt_magi_get);
        cudaEventRecord(evt_magi_get, smgr->stream(0));
        FMOE_SWE(smgr->stream(num_experts), evt_magi_get);
        cudaEventDestroy(evt_magi_get);

        for (long j = i * world_size; j < i * world_size + world_size; ++j) {
          if (receive_models[j]) {
            NCCL_SAFE_CALL(ncclSend((void *)send_params[send_params_idx].data_ptr<scalar_t>(),
                                    expert_size * sizeof(scalar_t), ncclChar, j % world_size, smgr->ncclcomm,
                                    smgr->stream(num_experts)));
          }
        }
        send_params_idx++;
      } else {
        // receive part
        if (receive_models[i * world_size + rank]) {
          NCCL_SAFE_CALL(ncclRecv((void *)receive_params[receive_params_idx].data_ptr<scalar_t>(),
                                  expert_size * sizeof(scalar_t), ncclChar, i / num_experts, smgr->ncclcomm,
                                  smgr->stream(num_experts)));
        }
      }
      NCCL_SAFE_CALL(ncclGroupEnd());
      if (i / num_experts != rank && receive_models[i * world_size + rank]) {
        cudaEventCreate(evt_magi_receive + receive_params_idx);
        cudaEventRecord(evt_magi_receive[receive_params_idx], smgr->stream(num_experts));
        receive_params_idx++;
      }
    }
    if (magi_profile_flag)
      cudaEventRecord(magi_stime_end[i], smgr->stream(num_experts));
  }

  // C_0 ... C_n
  for (long step = 0; step < n_groups; ++step) {
    if (magi_profile_flag)
      cudaEventRecord(ctime_launch[step], smgr->stream(0));
    FMOE_SWE(smgr->stream(0), input_ready[step]);
    FMOE_SWE(smgr->torchStream(), input_ready[step]);
    if (magi_profile_flag)
      cudaEventRecord(ctime_start[step], smgr->stream(0));
    for (int ei = 0; ei < num_experts; ++ei) {
      GEN_BASE(step);
      long offset = global_ptr[ei * world_size + from_base];
      long micro_batch_size = global_ptr[ei * world_size + (from_base + pipeline_gran)] - offset;
      computeFn(forward_fn, device, global_input_buf, global_output_buf, (long)ei, step * num_experts + ei, offset,
                micro_batch_size, d_model, 0, smgr);
    }
    cudaEventRecord(output_ready[step], smgr->stream(0));
    cudaEventRecord(output_torch_ready[step], smgr->torchStream());
  }
  // Compute over shadowed experts
  // for (long i = 0, si = 0; i < world_size * num_experts; ++i) {

  //     if (stored_models[i]) {
  //         if (magi_profile_flag) cudaEventRecord(magi_ctime_launch[i],
  //         smgr->stream(0)); FMOE_SWE(smgr->stream(0), evt_shadow[si]);
  //         FMOE_SWE(smgr->torchStream(), evt_shadow[si]);
  //         if (magi_profile_flag) cudaEventRecord(magi_ctime_start[i],
  //         smgr->stream(0)); stash_fn(send_params[si], si, 0); // always put
  //         shadowed expert at first, so expert_idx = 0 save shadow_expert in
  //         expert0 ,original expert is put in expert_param_stash long offset =
  //         local_ptr[i]; long micro_batch_size = local_expert_count[i];
  //         computeFn(forward_fn, device,
  //                 input_buf, output_buf,
  //                 0, n_groups * num_experts + si, offset, micro_batch_size,
  //                 d_model, smgr);
  //         ++si;
  //         if (magi_profile_flag) cudaEventRecord(magi_ctime_end[i],
  //         smgr->stream(0));
  //     }

  // }
  // pop_fn(0);

  // Compute over magi experts (receive part)
  for (long i = 0, receive_expert_cnt = 0; i < world_size * num_experts; ++i) {
    if (send_models[i]) {
      if (receive_models[i * world_size + rank]) {
        if (magi_profile_flag)
          cudaEventRecord(magi_ctime_launch[receive_expert_cnt], smgr->stream(0));
        FMOE_SWE(smgr->stream(0), evt_magi_receive[receive_expert_cnt]);
        FMOE_SWE(smgr->torchStream(), evt_magi_receive[receive_expert_cnt]);
        if (magi_profile_flag)
          cudaEventRecord(magi_ctime_start[receive_expert_cnt], smgr->stream(0));
        push_magi_expert_fn(receive_params[receive_expert_cnt], i);
        long offset = local_ptr[i];
        long micro_batch_size = local_expert_count[i];
        computeFn(forward_fn, device, input_buf, output_buf, i, n_groups * num_experts + receive_expert_cnt, offset,
                  micro_batch_size, d_model, 1, smgr);
        if (magi_profile_flag)
          cudaEventRecord(magi_ctime_end[receive_expert_cnt], smgr->stream(0));
        ++receive_expert_cnt;
      }
    }
  }

  // Compute over magi experts (keep part)
  for (long i = 0, keep_expert_cnt = 0; i < world_size * num_experts; ++i) {
    if (magi_profile_flag)
      cudaEventRecord(keep_start[i], smgr->stream(0));
    if (keep_models[num_experts * world_size * rank + i] > 0) {
      // if (is_magi_expert_exist(is_magi_expert_exist_fn, rank, i)) {
      long offset = local_ptr[i];
      long micro_batch_size = local_expert_count[i];
      computeFn(forward_fn, device, input_buf, output_buf, i,
                n_groups * num_experts + receive_params.size() + keep_expert_cnt, offset, micro_batch_size, d_model, 1,
                smgr);
      ++keep_expert_cnt;
    }
    if (magi_profile_flag)
      cudaEventRecord(keep_end[i], smgr->stream(0));
  }
  // pop_fn(0);

  // R_0 ... R_n
  for (long step = 0; step < n_groups; ++step) {
    if (magi_profile_flag)
      cudaEventRecord(rtime_launch[step], smgr->stream(num_experts));
    FMOE_SWE(smgr->stream(num_experts), output_ready[step]);
    FMOE_SWE(smgr->stream(num_experts), output_torch_ready[step]);
    if (magi_profile_flag)
      cudaEventRecord(rtime_start[step], smgr->stream(num_experts));
    for (int ei = 0; ei < num_experts; ++ei) {
      GEN_BASE(step);
      NCCL_SAFE_CALL(ncclGroupStart());
      for (int j = 0; j < pipeline_gran; ++j) {
        int rank_send = j + from_base;
        int rank_recv = j + to_base;
        GEN_IDX;
        exchangeWith(global_output_buf + global_ptr[gidx_send] * d_model,
                     global_expert_count[idx_send] * !receive_models[idx_self * world_size + rank_send] *
                         !(keep_models[num_experts * world_size * rank_send + idx_self] > 0),
                     //  !is_magi_expert_exist(is_magi_expert_exist_fn, rank_send, idx_self),
                     rank_send,

                     output_buf + local_ptr[idx_recv] * d_model,
                     local_expert_count[idx_recv] * !receive_models[idx_recv * world_size + rank] *
                         !(keep_models[num_experts * world_size * rank + idx_recv] > 0),
                     //  !is_magi_expert_exist(is_magi_expert_exist_fn, rank, idx_recv),
                     rank_recv, d_model, smgr->stream(num_experts), smgr->ncclcomm);
      }
      NCCL_SAFE_CALL(ncclGroupEnd());
    }
    if (magi_profile_flag)
      cudaEventRecord(rtime_end[step], smgr->stream(num_experts));
  }

  if (magi_profile_flag) {
    float milliseconds = 0.0f, stime = 0.0f, ctime_wait = 0.0f, ctime = 0.0f, rtime_wait = 0.0f, rtime = 0.0f,
          magi_stime = 0.0f, magi_ctime_wait = 0.0f, magi_ctime = 0.0f, keep_ctime = 0.0f;

    for (int step = 0; step < n_groups; ++step) {
      cudaEventSynchronize(input_ready[step]);
      cudaEventElapsedTime(&milliseconds, stime_start[step], input_ready[step]);
      stime += milliseconds;
      cudaEventSynchronize(ctime_start[step]);
      cudaEventElapsedTime(&milliseconds, ctime_launch[step], ctime_start[step]);
      ctime_wait += milliseconds;
      cudaEventSynchronize(output_ready[step]);
      cudaEventElapsedTime(&milliseconds, ctime_start[step], output_ready[step]);
      ctime += milliseconds;
      cudaEventSynchronize(rtime_start[step]);
      cudaEventElapsedTime(&milliseconds, rtime_launch[step], rtime_start[step]);
      rtime_wait += milliseconds;
      cudaEventSynchronize(rtime_end[step]);
      cudaEventElapsedTime(&milliseconds, rtime_start[step], rtime_end[step]);
      rtime += milliseconds;
    }

    for (int i = 0; i < world_size * num_experts; ++i) {
      cudaEventSynchronize(magi_stime_end[i]);
      cudaEventElapsedTime(&milliseconds, magi_stime_start[i], magi_stime_end[i]);
      magi_stime += milliseconds;
      cudaEventSynchronize(keep_end[i]);
      cudaEventElapsedTime(&milliseconds, keep_start[i], keep_end[i]);
      keep_ctime += milliseconds;
    }
    for (unsigned i = 0; i < receive_params.size(); ++i) {
      cudaEventSynchronize(magi_ctime_start[i]);
      cudaEventElapsedTime(&milliseconds, magi_ctime_launch[i], magi_ctime_start[i]);
      magi_ctime_wait += milliseconds;
      cudaEventSynchronize(magi_ctime_end[i]);
      cudaEventElapsedTime(&milliseconds, magi_ctime_start[i], magi_ctime_end[i]);
      magi_ctime += milliseconds;
    }
    record_layer_time_fn(stime, ctime, ctime_wait, rtime, rtime_wait,
                         magi_stime, magi_ctime, magi_ctime_wait, keep_ctime);
  }

  smgr->sync(num_experts + 1);

  delete[] local_ptr;
  delete[] global_ptr;

  checkCudaErrors(cudaGetLastError());
  for (long i = 0; i < n_groups; ++i) {
    cudaEventDestroy(input_ready[i]);
    cudaEventDestroy(output_ready[i]);
    cudaEventDestroy(output_torch_ready[i]);

    cudaEventDestroy(stime_start[i]);
    cudaEventDestroy(ctime_launch[i]);
    cudaEventDestroy(ctime_start[i]);
    cudaEventDestroy(rtime_launch[i]);
    cudaEventDestroy(rtime_start[i]);
    cudaEventDestroy(rtime_end[i]);
  }

  for (long i = 0; i < world_size * num_experts; ++i) {
    cudaEventDestroy(magi_stime_start[i]);
    cudaEventDestroy(magi_stime_end[i]);
    cudaEventDestroy(magi_ctime_launch[i]);
    cudaEventDestroy(magi_ctime_start[i]);
    cudaEventDestroy(magi_ctime_end[i]);
    cudaEventDestroy(keep_start[i]);
    cudaEventDestroy(keep_end[i]);
  }

  delete[] input_ready;
  delete[] output_ready;
  delete[] output_torch_ready;

  delete[] stime_start;
  delete[] ctime_launch;
  delete[] ctime_start;
  delete[] rtime_launch;
  delete[] rtime_start;
  delete[] rtime_end;

  delete[] magi_stime_start;
  delete[] magi_stime_end;
  delete[] magi_ctime_launch;
  delete[] magi_ctime_start;
  delete[] magi_ctime_end;
  delete[] keep_start;
  delete[] keep_end;
}

template <typename scalar_t>
void fmoe_cuda_fused_backward_impl(
    py::function backward_fn,
    py::function record_layer_time_fn,
    py::function collect_fn,
    py::function set_grad_fn, c10::Device device,

    scalar_t *grad_out, scalar_t *global_grad_out, scalar_t *global_grad_in,
    scalar_t *grad_in,

    const long *local_expert_count, const long *global_expert_count,
    const bool *send_models, const bool *receive_models, const int *keep_models,
    long d_model, long num_experts,
    long rank, long world_size, long pipeline_gran, bool magi_profile_flag, CudaStreamManager *smgr) {
  smgr->syncTorch();

  int *local_ptr = new int[num_experts * world_size + 1];
  int *global_ptr = new int[num_experts * world_size + 1];

  computePtrs(num_experts, rank, world_size, local_expert_count, global_expert_count, send_models, receive_models, keep_models, local_ptr, global_ptr);
  if (pipeline_gran > world_size) {
    pipeline_gran = world_size;
  }
  long n_groups = world_size / pipeline_gran;
  long group_rank = rank / pipeline_gran;

  cudaEvent_t *input_ready = new cudaEvent_t[n_groups];
  cudaEvent_t *output_ready = new cudaEvent_t[n_groups];
  cudaEvent_t *output_torch_ready = new cudaEvent_t[n_groups];

  cudaEvent_t *stime_start = new cudaEvent_t[n_groups];
  cudaEvent_t *ctime_launch = new cudaEvent_t[n_groups];
  cudaEvent_t *ctime_start = new cudaEvent_t[n_groups];
  cudaEvent_t *rtime_launch = new cudaEvent_t[n_groups];
  cudaEvent_t *rtime_start = new cudaEvent_t[n_groups];
  cudaEvent_t *rtime_end = new cudaEvent_t[n_groups];
  for (long i = 0; i < n_groups; ++i) {
    cudaEventCreate(input_ready + i);
    cudaEventCreate(output_ready + i);
    cudaEventCreate(output_torch_ready + i);
    
    cudaEventCreate(stime_start + i);
    cudaEventCreate(ctime_launch + i);
    cudaEventCreate(ctime_start + i);
    cudaEventCreate(rtime_launch + i);
    cudaEventCreate(rtime_start + i);
    cudaEventCreate(rtime_end + i);
  }

  cudaEvent_t *magi_start = new cudaEvent_t[world_size * num_experts];
  cudaEvent_t *magi_end = new cudaEvent_t[world_size * num_experts];
  cudaEvent_t *magi_reduce_start = new cudaEvent_t[world_size * num_experts];
  cudaEvent_t *keep_start = new cudaEvent_t[world_size * num_experts];
  cudaEvent_t *keep_end = new cudaEvent_t[world_size * num_experts];
  cudaEvent_t *keep_reduce_start = new cudaEvent_t[world_size * num_experts];
  cudaEvent_t *set_gradients_start = new cudaEvent_t[world_size * num_experts];
  cudaEvent_t *set_gradients_end = new cudaEvent_t[world_size * num_experts];

  for (long i = 0; i < world_size * num_experts; ++i) {
    cudaEventCreate(magi_start + i);
    cudaEventCreate(magi_end + i);
    cudaEventCreate(magi_reduce_start + i);
    cudaEventCreate(keep_start + i);
    cudaEventCreate(keep_end + i);
    cudaEventCreate(keep_reduce_start + i);
    cudaEventCreate(set_gradients_start + i);
    cudaEventCreate(set_gradients_end + i);
  }

  // S_0 ... S_n
  for (long step = 0; step < n_groups; ++step) {
    if (magi_profile_flag)
      cudaEventRecord(stime_start[step], smgr->stream(num_experts));
    for (int ei = 0; ei < num_experts; ++ei) {
      GEN_BASE(step);
      NCCL_SAFE_CALL(ncclGroupStart());
      for (int j = 0; j < pipeline_gran; ++j) {
        int rank_send = j + to_base;
        int rank_recv = j + from_base;
        GEN_IDX;
        exchangeWith(grad_out + local_ptr[idx_send] * d_model,
                     local_expert_count[idx_send] * !receive_models[idx_send * world_size + rank] *
                         !(keep_models[num_experts * world_size * rank + idx_send] > 0),
                     //  !is_magi_expert_exist(is_magi_expert_exist_fn, rank, idx_send),
                     rank_send,

                     global_grad_out + global_ptr[gidx_recv] * d_model,
                     global_expert_count[idx_recv] * !receive_models[idx_self * world_size + rank_recv] *
                         !(keep_models[num_experts * world_size * rank_recv + idx_self] > 0),
                     //  !is_magi_expert_exist(is_magi_expert_exist_fn, rank_recv, idx_self),
                     rank_recv,

                     d_model, smgr->stream(num_experts), smgr->ncclcomm);
      }
      NCCL_SAFE_CALL(ncclGroupEnd());
    }
    cudaEventRecord(input_ready[step], smgr->stream(num_experts));
  }

  // // Shadowed experts backward and reduce
  // cudaEvent_t *evt_reduce = new cudaEvent_t[num_experts];
  // for (long i = 0, si = 0; i < world_size * num_experts; ++i) {
  //     if (send_models[i]) {
  //         stash_fn(si, 0);
  //         long offset = local_ptr[i];
  //         long micro_batch_size = local_expert_count[i];
  //         computeFn(backward_fn, device,
  //                 grad_out, grad_in,
  //                 0, n_groups * num_experts + si, offset, micro_batch_size,
  //                 d_model, smgr);
  //         collect_fn(si, i / num_experts, 0);
  //         if (i / num_experts == rank) {
  //             cudaEventCreate(evt_reduce + i % num_experts);
  //             cudaEventRecord(evt_reduce[i % num_experts], smgr->stream(0));
  //         }
  //         ++si;
  //     }
  // }
  // pop_fn(0);

  long receive_expert_num = 0;
  // magi experts backward and reduce (receive part)
  cudaEvent_t *evt_receive_reduce = new cudaEvent_t[num_experts * world_size];
  for (long i = 0, receive_expert_cnt = 0; i < world_size * num_experts; ++i) {
    if (send_models[i]) {
      if (receive_models[i * world_size + rank]) {
        long offset = local_ptr[i];
        long micro_batch_size = local_expert_count[i];
        if (magi_profile_flag)
          cudaEventRecord(magi_start[receive_expert_cnt], smgr->stream(0));
        computeFn(backward_fn, device, grad_out, grad_in, i, n_groups * num_experts + receive_expert_cnt, offset,
                  micro_batch_size, d_model, 1, smgr);
        if (magi_profile_flag)
          cudaEventRecord(magi_end[receive_expert_cnt], smgr->stream(0));
        if (magi_profile_flag)
          cudaEventRecord(magi_reduce_start[i], smgr->stream(0));
        collect_fn(i, 1, 0);
        ++receive_expert_cnt;
        ++receive_expert_num;
      } else {
        if (magi_profile_flag)
          cudaEventRecord(magi_reduce_start[i], smgr->stream(0));
        collect_fn(i, 0, 0);
      }
      cudaEventCreate(evt_receive_reduce + i);
      cudaEventRecord(evt_receive_reduce[i], smgr->stream(0));
    }
  }

  long keep_expert_num = 0;
  // magi experts backward and reduce (keep part)
  cudaEvent_t *evt_keep_reduce = new cudaEvent_t[num_experts * world_size];
  for (long i = 0, keep_expert_cnt = 0; i < world_size * num_experts; ++i) {
    // torch::Tensor bool_tensor = torch::zeros({1}, torch::kBool);
    // is_global_magi_expert_exist_fn(bool_tensor, i);
    // bool global_exist_flag = bool_tensor.data_ptr<bool>()[0];
    if (is_global_keep_expert_exist(num_experts, world_size, keep_models)) {
      if (keep_models[num_experts * world_size * rank + i] > 0) {
        // if (is_magi_expert_exist(is_magi_expert_exist_fn, rank, i)) {
        long offset = local_ptr[i];
        long micro_batch_size = local_expert_count[i];
        if (magi_profile_flag)
          cudaEventRecord(keep_start[keep_expert_cnt], smgr->stream(0));
        computeFn(backward_fn, device, grad_out, grad_in, i,
                  n_groups * num_experts + receive_expert_num + keep_expert_cnt, offset, micro_batch_size, d_model, 1,
                  smgr);
        if (magi_profile_flag)
          cudaEventRecord(keep_end[keep_expert_cnt], smgr->stream(0));
        if (magi_profile_flag)
          cudaEventRecord(keep_reduce_start[i], smgr->stream(0));
        collect_fn(i, 1, 1);
        ++keep_expert_cnt;
        ++keep_expert_num;
      } else {
        if (magi_profile_flag)
          cudaEventRecord(keep_reduce_start[i], smgr->stream(0));
        collect_fn(i, 0, 1);
      }
      cudaEventCreate(evt_keep_reduce + i);
      cudaEventRecord(evt_keep_reduce[i], smgr->stream(0));
    }
  }
  // pop_fn(0);

  // C_0 ... C_n
  for (long step = 0; step < n_groups; ++step) {
    if (magi_profile_flag)
      cudaEventRecord(ctime_launch[step], smgr->stream(0));
    FMOE_SWE(smgr->stream(0), input_ready[step]);
    FMOE_SWE(smgr->torchStream(), input_ready[step]);
    if (magi_profile_flag)
      cudaEventRecord(ctime_start[step], smgr->stream(0));
    for (int ei = 0; ei < num_experts; ++ei) {
      GEN_BASE(step);
      long offset = global_ptr[ei * world_size + from_base];
      long micro_batch_size = global_ptr[ei * world_size + (from_base + pipeline_gran)] - offset;
      computeFn(backward_fn, device, global_grad_out, global_grad_in, (long)ei, step * num_experts + ei, offset,
                micro_batch_size, d_model, 0, smgr);
    }
    cudaEventRecord(output_ready[step], smgr->stream(0));
    cudaEventRecord(output_torch_ready[step], smgr->torchStream());
  }

  // Collect gradients for magi experts
  for (long i = 0; i < world_size * num_experts; ++i) {
    if (magi_profile_flag)
      cudaEventRecord(set_gradients_start[i], smgr->stream(0));
    if (send_models[i]) {
      // receive part
      FMOE_SWE(smgr->torchStream(), evt_receive_reduce[i]);
      if (receive_models[i * world_size + rank]) {
        set_grad_fn(i, 0);
      } else if (i / num_experts == rank) {
        set_grad_fn(i, 2);
      }
      // } else if (is_magi_expert_exist(is_magi_expert_exist_fn, rank, i)) {
    } else if (keep_models[num_experts * world_size * rank + i] > 0) {
      // keep part
      FMOE_SWE(smgr->torchStream(), evt_keep_reduce[i]);
      set_grad_fn(i, 1);
    }
    if (magi_profile_flag)
      cudaEventRecord(set_gradients_end[i], smgr->stream(0));
  }

  // R_0 ... R_n
  for (long step = 0; step < n_groups; ++step) {
    if (magi_profile_flag)
      cudaEventRecord(rtime_launch[step], smgr->stream(num_experts));
    FMOE_SWE(smgr->stream(num_experts), output_ready[step]);
    FMOE_SWE(smgr->stream(num_experts), output_torch_ready[step]);
    if (magi_profile_flag)
      cudaEventRecord(rtime_start[step], smgr->stream(num_experts));
    for (int ei = 0; ei < num_experts; ++ei) {
      GEN_BASE(step);
      NCCL_SAFE_CALL(ncclGroupStart());
      for (int j = 0; j < pipeline_gran; ++j) {
        int rank_send = j + from_base;
        int rank_recv = j + to_base;
        GEN_IDX;
        exchangeWith(global_grad_in + global_ptr[gidx_send] * d_model,
                     global_expert_count[idx_send] * !receive_models[idx_self * world_size + rank_send] *
                         !(keep_models[num_experts * world_size * rank_send + idx_self] > 0),
                     //  !is_magi_expert_exist(is_magi_expert_exist_fn, rank_send, idx_self),
                     rank_send,

                     grad_in + local_ptr[idx_recv] * d_model,
                     local_expert_count[idx_recv] * !receive_models[idx_recv * world_size + rank] *
                         !(keep_models[num_experts * world_size * rank + idx_recv] > 0),
                     //  !is_magi_expert_exist(is_magi_expert_exist_fn, rank, idx_recv),
                     rank_recv, d_model, smgr->stream(num_experts), smgr->ncclcomm);
      }
      NCCL_SAFE_CALL(ncclGroupEnd());
    }
    if (magi_profile_flag)
      cudaEventRecord(rtime_end[step], smgr->stream(num_experts));
  }

  if (magi_profile_flag) {
    float milliseconds = 0.0f, stime = 0.0f, ctime_wait = 0.0f, ctime = 0.0f, rtime_wait = 0.0f, rtime = 0.0f,
          magi_ctime = 0.0f, magi_reduce = 0.0f, keep_ctime = 0.0f, keep_reduce = 0.0f, set_gradients = 0.0f;

    for (int step = 0; step < n_groups; ++step) {
      cudaEventSynchronize(input_ready[step]);
      cudaEventElapsedTime(&milliseconds, stime_start[step], input_ready[step]);
      stime += milliseconds;
      cudaEventSynchronize(ctime_start[step]);
      cudaEventElapsedTime(&milliseconds, ctime_launch[step], ctime_start[step]);
      ctime_wait += milliseconds;
      cudaEventSynchronize(output_ready[step]);
      cudaEventElapsedTime(&milliseconds, ctime_start[step], output_ready[step]);
      ctime += milliseconds;
      cudaEventSynchronize(rtime_start[step]);
      cudaEventElapsedTime(&milliseconds, rtime_launch[step], rtime_start[step]);
      rtime_wait += milliseconds;
      cudaEventSynchronize(rtime_end[step]);
      cudaEventElapsedTime(&milliseconds, rtime_start[step], rtime_end[step]);
      rtime += milliseconds;
    }

    for (int i = 0; i < receive_expert_num; ++i) {
      cudaEventSynchronize(magi_end[i]);
      cudaEventElapsedTime(&milliseconds, magi_start[i], magi_end[i]);
      magi_ctime += milliseconds;
    }
    for (int i = 0; i < keep_expert_num; ++i) {
      cudaEventSynchronize(keep_end[i]);
      cudaEventElapsedTime(&milliseconds, keep_start[i], keep_end[i]);
      keep_ctime += milliseconds;
    }

    for (int i = 0; i < world_size * num_experts; ++i) {
      if (send_models[i]) {
        cudaEventSynchronize(evt_receive_reduce[i]);
        cudaEventElapsedTime(&milliseconds, magi_reduce_start[i], evt_receive_reduce[i]);
        magi_reduce += milliseconds;
      }
      if (is_global_keep_expert_exist(num_experts, world_size, keep_models)) {
        cudaEventSynchronize(evt_keep_reduce[i]);
        cudaEventElapsedTime(&milliseconds, keep_reduce_start[i], evt_keep_reduce[i]);
        keep_reduce += milliseconds;
      }
      cudaEventSynchronize(set_gradients_end[i]);
      cudaEventElapsedTime(&milliseconds, set_gradients_start[i], set_gradients_end[i]);
      set_gradients += milliseconds;
    }
    record_layer_time_fn(stime, ctime, ctime_wait, rtime, rtime_wait,
                         magi_ctime, magi_reduce, keep_ctime, keep_reduce,
                         set_gradients);
  }

  smgr->sync(num_experts + 1);
  checkCudaErrors(cudaGetLastError());

  delete[] local_ptr;
  delete[] global_ptr;
  checkCudaErrors(cudaGetLastError());
  for (long i = 0; i < n_groups; ++i) {
    cudaEventDestroy(input_ready[i]);
    cudaEventDestroy(output_ready[i]);
    cudaEventDestroy(output_torch_ready[i]);

    cudaEventDestroy(stime_start[i]);
    cudaEventDestroy(ctime_launch[i]);
    cudaEventDestroy(ctime_start[i]);
    cudaEventDestroy(rtime_launch[i]);
    cudaEventDestroy(rtime_start[i]);
    cudaEventDestroy(rtime_end[i]);
  }
  delete[] input_ready;
  delete[] output_ready;
  delete[] output_torch_ready;

  delete[] stime_start;
  delete[] ctime_launch;
  delete[] ctime_start;
  delete[] rtime_launch;
  delete[] rtime_start;
  delete[] rtime_end;

  for (long i = 0; i < receive_expert_num; i++) {
    cudaEventDestroy(magi_start[i]);
    cudaEventDestroy(magi_end[i]);
  }
  for (long i = 0; i < keep_expert_num; i++) {
    cudaEventDestroy(keep_start[i]);
    cudaEventDestroy(keep_end[i]);
  }
  for (long i = 0; i < num_experts * world_size; ++i) {
    if (send_models[i]) {
      cudaEventDestroy(evt_receive_reduce[i]);
      cudaEventDestroy(magi_reduce_start[i]);
    }
    if (is_global_keep_expert_exist(num_experts, world_size, keep_models)) {
      // if (is_magi_expert_exist(is_magi_expert_exist_fn, rank, i)) {
      cudaEventDestroy(evt_keep_reduce[i]);
      cudaEventDestroy(keep_reduce_start[i]);
    }
  }
  for (long i = 0; i < world_size * num_experts; i++) {
    cudaEventDestroy(set_gradients_start[i]);
    cudaEventDestroy(set_gradients_end[i]);
  }
  delete[] evt_keep_reduce;
  delete[] evt_receive_reduce;

  delete[] magi_start;
  delete[] magi_end;
  delete[] magi_reduce_start;
  delete[] keep_start;
  delete[] keep_end;
  delete[] keep_reduce_start;
  delete[] set_gradients_start;
  delete[] set_gradients_end;
}

#endif // SMART_SCHEDULE_H
