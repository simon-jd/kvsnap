#include "cuda_runtime_api.h"
#include <cstdlib>
#include <cstring>
struct FakeEvent { bool recorded=false; };
static bool fail_record=false;
static bool fail_sync=false;
static int pending=0;
static cudaStream_t last_copy_stream=nullptr;
extern "C" {
const char* cudaGetErrorString(cudaError_t) { return "injected CUDA error"; }
cudaError_t cudaSetDevice(int) { return cudaSuccess; }
cudaError_t cudaHostAlloc(void** p,std::size_t size,unsigned) { *p=std::malloc(size); return *p?cudaSuccess:cudaErrorUnknown; }
cudaError_t cudaFreeHost(void* p) { std::free(p); return cudaSuccess; }
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* p,unsigned) { *p=std::malloc(1); return cudaSuccess; }
cudaError_t cudaStreamSynchronize(cudaStream_t) { if(fail_sync) { fail_sync=false; return cudaErrorUnknown; } pending=0; return cudaSuccess; }
cudaError_t cudaStreamQuery(cudaStream_t) { return pending?cudaErrorNotReady:cudaSuccess; }
cudaError_t cudaStreamDestroy(cudaStream_t p) { std::free(p); return cudaSuccess; }
cudaError_t cudaStreamWaitEvent(cudaStream_t,cudaEvent_t,unsigned) { return cudaSuccess; }
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* p,unsigned) { *p=new FakeEvent(); return cudaSuccess; }
cudaError_t cudaEventRecord(cudaEvent_t p,cudaStream_t) { if(fail_record) { fail_record=false; return cudaErrorUnknown; } static_cast<FakeEvent*>(p)->recorded=true; return cudaSuccess; }
cudaError_t cudaEventSynchronize(cudaEvent_t p) { if(static_cast<FakeEvent*>(p)->recorded) pending=0; return cudaSuccess; }
cudaError_t cudaEventQuery(cudaEvent_t p) { return static_cast<FakeEvent*>(p)->recorded && pending?cudaErrorNotReady:cudaSuccess; }
cudaError_t cudaEventDestroy(cudaEvent_t p) { delete static_cast<FakeEvent*>(p); return cudaSuccess; }
cudaError_t cudaMemcpyAsync(void* to,const void* from,std::size_t size,int,cudaStream_t stream) { last_copy_stream=stream; std::memcpy(to,from,size); pending++; return cudaSuccess; }
cudaError_t cudaMemcpy2DAsync(void* to,std::size_t dpitch,const void* from,std::size_t spitch,std::size_t width,std::size_t height,int,cudaStream_t stream) { last_copy_stream=stream; for(std::size_t r=0;r<height;r++) std::memcpy(static_cast<char*>(to)+r*dpitch,static_cast<const char*>(from)+r*spitch,width); pending++; return cudaSuccess; }
void kvsnap_test_fail_next_record() { fail_record=true; }
void kvsnap_test_fail_next_stream_sync() { fail_sync=true; }
int kvsnap_test_pending() { return pending; }
std::uintptr_t kvsnap_test_last_copy_stream() { return reinterpret_cast<std::uintptr_t>(last_copy_stream); }
}
