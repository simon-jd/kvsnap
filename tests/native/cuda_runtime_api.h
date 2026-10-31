// Minimal test-only ABI shim. This is not a CUDA implementation or GPU proof.
#pragma once
#include <cstddef>
#include <cstdint>
typedef int cudaError_t;
typedef void* cudaStream_t;
typedef void* cudaEvent_t;
constexpr int cudaSuccess=0, cudaErrorNotReady=34, cudaErrorUnknown=999;
constexpr int cudaHostAllocPortable=1, cudaStreamNonBlocking=1, cudaEventDisableTiming=2;
constexpr int cudaMemcpyDeviceToHost=2, cudaMemcpyHostToDevice=1;
extern "C" {
const char* cudaGetErrorString(cudaError_t);
cudaError_t cudaSetDevice(int);
cudaError_t cudaHostAlloc(void**,std::size_t,unsigned);
cudaError_t cudaFreeHost(void*);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t*,unsigned);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaStreamQuery(cudaStream_t);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaStreamWaitEvent(cudaStream_t,cudaEvent_t,unsigned);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t*,unsigned);
cudaError_t cudaEventRecord(cudaEvent_t,cudaStream_t);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaEventQuery(cudaEvent_t);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaMemcpyAsync(void*,const void*,std::size_t,int,cudaStream_t);
cudaError_t cudaMemcpy2DAsync(void*,std::size_t,const void*,std::size_t,std::size_t,std::size_t,int,cudaStream_t);
void kvsnap_test_fail_next_record();
void kvsnap_test_fail_next_stream_sync();
int kvsnap_test_pending();
std::uintptr_t kvsnap_test_last_copy_stream();
}
