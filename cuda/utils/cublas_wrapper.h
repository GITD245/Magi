#ifndef CUBLAS_WRAPPER_H
#define CUBLAS_WRAPPER_H
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cublas_v2.h>

inline cublasStatus_t
cublasXgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                   cublasOperation_t transb, int m, int n, int k,
                   const float *alpha, const float *Aarray[], int lda,
                   const float *Barray[], int ldb, const float *beta,
                   float *Carray[], int ldc, int batchCount) {
  return cublasSgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda,
                            Barray, ldb, beta, Carray, ldc, batchCount);
}

inline cublasStatus_t
cublasXgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                   cublasOperation_t transb, int m, int n, int k,
                   const double *alpha, const double *Aarray[], int lda,
                   const double *Barray[], int ldb, const double *beta,
                   double *Carray[], int ldc, int batchCount) {
  return cublasDgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda,
                            Barray, ldb, beta, Carray, ldc, batchCount);
}

inline cublasStatus_t
cublasXgemmBatched(cublasHandle_t handle, cublasOperation_t transa,
                   cublasOperation_t transb, int m, int n, int k,
                   const __half *alpha, const __half *Aarray[], int lda,
                   const __half *Barray[], int ldb, const __half *beta,
                   __half *Carray[], int ldc, int batchCount) {
#ifdef FMOE_USE_HIP
  return rocblas_hgemm_batched(
      handle, transa, transb, m, n, k, (const rocblas_half *)alpha,
      (const rocblas_half *const *)Aarray, lda,
      (const rocblas_half *const *)Barray, ldb, (const rocblas_half *)beta,
      (rocblas_half *const *)Carray, ldc, batchCount);
#else
  return cublasHgemmBatched(handle, transa, transb, m, n, k, alpha, Aarray, lda,
                            Barray, ldb, beta, Carray, ldc, batchCount);
#endif
}

inline cublasStatus_t cublasXgemm(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb, int m, int n, int k,
                                  const float *alpha, const float *A, int lda,
                                  const float *B, int ldb, const float *beta,
                                  float *C, int ldc) {
  return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}

inline cublasStatus_t cublasXgemm(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb, int m, int n, int k,
                                  const double *alpha, const double *A, int lda,
                                  const double *B, int ldb, const double *beta,
                                  double *C, int ldc) {
  return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
}

inline cublasStatus_t cublasXgemm(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb, int m, int n, int k,
                                  const __half *alpha, const __half *A, int lda,
                                  const __half *B, int ldb, const __half *beta,
                                  __half *C, int ldc) {
#ifdef FMOE_USE_HIP
  return rocblas_hgemm(handle, transa, transb, m, n, k,
                       (const rocblas_half *)alpha, (const rocblas_half *)A,
                       lda, (const rocblas_half *)B, ldb,
                       (const rocblas_half *)beta, (rocblas_half *)C, ldc);
#else
  return cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb,
                     beta, C, ldc);
#endif
}

inline cublasStatus_t cublasXgemm(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k, const c10::Half *alpha, const c10::Half *A, int lda,
    const c10::Half *B, int ldb, const c10::Half *beta, c10::Half *C, int ldc) {
#ifdef FMOE_USE_HIP
  return rocblas_hgemm(handle, transa, transb, m, n, k,
                       (const rocblas_half *)alpha, (const rocblas_half *)A,
                       lda, (const rocblas_half *)B, ldb,
                       (const rocblas_half *)beta, (rocblas_half *)C, ldc);
#else
  return cublasHgemm(handle, transa, transb, m, n, k, (const __half *)alpha,
                     (const __half *)A, lda, (const __half *)B, ldb,
                     (const __half *)beta, (__half *)C, ldc);
#endif
}

inline cublasStatus_t
cublasXgemm(cublasHandle_t handle, cublasOperation_t transa,
            cublasOperation_t transb, int m, int n, int k,
            const c10::BFloat16 *alpha, const c10::BFloat16 *A, int lda,
            const c10::BFloat16 *B, int ldb, const c10::BFloat16 *beta,
            c10::BFloat16 *C, int ldc) {
#ifdef FMOE_USE_HIP
  // TODO: Support bf16 for HIP
  assert(false);
#else
  const float alpha_fp32(*alpha), beta_fp32(*beta);
  return cublasSgemmEx(handle, transa, transb, m, n, k,
                       (const float *)&alpha_fp32, (const void *)A, CUDA_R_16BF,
                       lda, (const void *)B, CUDA_R_16BF, ldb,
                       (const float *)&beta_fp32, (void *)C, CUDA_R_16BF, ldc);
#endif
}
#endif // CUBLAS_WRAPPER_H
