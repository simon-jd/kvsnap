#include <cuda_runtime_api.h>
#include <algorithm>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>

namespace {
thread_local std::string last_error;
void check(cudaError_t e) {
    if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));
}
struct Allocation {
    size_t offset, size;
    cudaEvent_t event{};
    bool submitted{false};
    bool event_valid{false};
};
struct Pool {
    void* data{};
    size_t capacity{};
    int device{};
    cudaStream_t stream{};
    std::map<uint64_t, Allocation> live;
    uint64_t next{1};
    bool drained{false};
    std::mutex mutex;
    ~Pool() {
        cudaSetDevice(device);
        // If the device cannot prove completion, leak/quarantine host memory
        // until process teardown instead of exposing it to another allocation.
        if (!drained && stream && cudaStreamSynchronize(stream) != cudaSuccess) return;
        for (auto& entry : live) cudaEventDestroy(entry.second.event);
        if (stream) cudaStreamDestroy(stream);
        if (data) cudaFreeHost(data);
    }
};
template<typename F> int guard(F&& f) {
    try { f(); return 0; }
    catch (const std::exception& e) { last_error = e.what(); return -1; }
}
Allocation& allocation(Pool* p, uint64_t id) {
    auto it = p->live.find(id);
    if (it == p->live.end()) throw std::runtime_error("unknown pinned allocation");
    return it->second;
}
}
extern "C" const char* kvsnap_cuda_error() { return last_error.c_str(); }
// Bump whenever Pool/Allocation layout or cross-library calling semantics change.
extern "C" int kvsnap_pool_abi_version() { return 2; }
extern "C" void* kvsnap_pool_create(size_t capacity, int device) {
    try {
        if (!capacity) throw std::runtime_error("pool capacity must be positive");
        auto p = std::make_unique<Pool>(); p->capacity = capacity; p->device = device;
        check(cudaSetDevice(device));
        check(cudaHostAlloc(&p->data, capacity, cudaHostAllocPortable));
        check(cudaStreamCreateWithFlags(&p->stream, cudaStreamNonBlocking));
        return p.release();
    } catch (const std::exception& e) { last_error = e.what(); return nullptr; }
}
extern "C" void kvsnap_pool_destroy(Pool* p) { delete p; }
// The caller excludes concurrent API entries. Preserve the complete pool on an
// uncertain drain so Python also retains the tensors referenced by in-flight DMA.
extern "C" int kvsnap_pool_close_checked(Pool* p) {
    return guard([&] {
        if (!p) return;
        check(cudaSetDevice(p->device));
        check(cudaStreamSynchronize(p->stream));
        p->drained = true;
        delete p;
    });
}
// Borrowed diagnostic handle, not a transfer or ownership handoff. The caller
// must exclude pool close for the full lifetime of external event operations.
// No Pool/Allocation layout change: existing ABI 2 copy bridges remain valid.
extern "C" int kvsnap_pool_copy_stream(Pool* p, uintptr_t* stream) {
    return guard([&] {
        if (!p || !stream) throw std::runtime_error("copy stream requires a live pool and output");
        std::lock_guard<std::mutex> lock(p->mutex);
        check(cudaSetDevice(p->device));
        *stream = reinterpret_cast<uintptr_t>(p->stream);
    });
}
extern "C" int kvsnap_pool_reserve(Pool* p, size_t size, uint64_t* id, void** ptr) {
    return guard([&] {
        std::lock_guard<std::mutex> lock(p->mutex);
        if (!size || size > p->capacity) throw std::runtime_error("pinned capacity exhausted");
        size_t offset = 0;
        // First fit, allocation ids need not follow address order.
        std::map<size_t, size_t> ranges;
        for (const auto& entry : p->live) ranges.emplace(entry.second.offset, entry.second.size);
        for (const auto& range : ranges) {
            if (range.first >= offset && range.first - offset >= size) break;
            offset = range.first + range.second;
        }
        if (offset > p->capacity - size) throw std::runtime_error("pinned capacity exhausted");
        check(cudaSetDevice(p->device));
        Allocation a{offset, size}; check(cudaEventCreateWithFlags(&a.event, cudaEventDisableTiming));
        *id = p->next++; *ptr = static_cast<char*>(p->data) + offset;
        p->live.emplace(*id, a);
    });
}
// Copies are serialized on a dedicated stream. The producer stream is fenced
// before D2H; H2D users must wait for the event before reading/reusing bytes.
extern "C" int kvsnap_pool_copy(Pool* p, uint64_t id, size_t host_offset,
                              void* device_ptr, size_t size, int to_host,
                              uintptr_t producer_stream) {
    return guard([&] {
        std::lock_guard<std::mutex> lock(p->mutex);
        auto& a = allocation(p, id);
        if (host_offset > a.size || size > a.size - host_offset)
            throw std::runtime_error("copy outside pinned allocation");
        check(cudaSetDevice(p->device));
        if (to_host) {
            cudaEvent_t ready{}; check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
            auto rc = cudaEventRecord(ready, reinterpret_cast<cudaStream_t>(producer_stream));
            if (rc == cudaSuccess) rc = cudaStreamWaitEvent(p->stream, ready, 0);
            cudaEventDestroy(ready); check(rc);
        }
        void* host = static_cast<char*>(p->data) + a.offset + host_offset;
        // An enqueue or event-record error must not leave an older event as the
        // authority for this allocation. Drain the stream if recording fails.
        a.submitted = true;
        a.event_valid = false;
        check(cudaMemcpyAsync(to_host ? host : device_ptr, to_host ? device_ptr : host,
                              size, to_host ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice,
                              p->stream));
        check(cudaEventRecord(a.event, p->stream));
        a.event_valid = true;
    });
}
// Strided copy of `height` rows of `width` bytes in one cudaMemcpy2DAsync.
// A KV layer stores block b at device_ptr + b * device_pitch, while a pinned
// snapshot packs all layers of a block together, so one layer's run of
// consecutive blocks is a 2D copy with host_pitch = bytes per packed block.
// This replaces `height` separate cudaMemcpyAsync calls per layer.
extern "C" int kvsnap_pool_copy_2d(Pool* p, uint64_t id, size_t host_offset, size_t host_pitch,
                                 void* device_ptr, size_t device_pitch, size_t width, size_t height,
                                 int to_host, uintptr_t producer_stream) {
    return guard([&] {
        std::lock_guard<std::mutex> lock(p->mutex);
        auto& a = allocation(p, id);
        if (!width || !height || host_pitch < width || device_pitch < width)
            throw std::runtime_error("invalid 2D copy geometry");
        const size_t span = (height - 1) * host_pitch + width;
        if (host_offset > a.size || span > a.size - host_offset)
            throw std::runtime_error("2D copy outside pinned allocation");
        check(cudaSetDevice(p->device));
        if (to_host) {
            cudaEvent_t ready{}; check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming));
            auto rc = cudaEventRecord(ready, reinterpret_cast<cudaStream_t>(producer_stream));
            if (rc == cudaSuccess) rc = cudaStreamWaitEvent(p->stream, ready, 0);
            cudaEventDestroy(ready); check(rc);
        }
        char* host = static_cast<char*>(p->data) + a.offset + host_offset;
        a.submitted = true;
        a.event_valid = false;
        if (to_host)
            check(cudaMemcpy2DAsync(host, host_pitch, device_ptr, device_pitch, width, height,
                                    cudaMemcpyDeviceToHost, p->stream));
        else
            check(cudaMemcpy2DAsync(device_ptr, device_pitch, host, host_pitch, width, height,
                                    cudaMemcpyHostToDevice, p->stream));
        check(cudaEventRecord(a.event, p->stream));
        a.event_valid = true;
    });
}
extern "C" int kvsnap_pool_wait(Pool* p, uint64_t id) {
    return guard([&] {
        std::lock_guard<std::mutex> lock(p->mutex);
        auto& a = allocation(p, id); check(cudaSetDevice(p->device));
        if (a.submitted) check(a.event_valid ? cudaEventSynchronize(a.event)
                                             : cudaStreamSynchronize(p->stream));
    });
}
extern "C" int kvsnap_pool_ready(Pool* p, uint64_t id) {
    int ready = 0;
    int rc = guard([&] {
        std::lock_guard<std::mutex> lock(p->mutex);
        auto& a = allocation(p, id); check(cudaSetDevice(p->device));
        if (!a.submitted) { ready = 1; return; }
        auto result = a.event_valid ? cudaEventQuery(a.event) : cudaStreamQuery(p->stream);
        if (result == cudaSuccess) ready = 1;
        else if (result != cudaErrorNotReady) check(result);
    });
    return rc < 0 ? -1 : ready;
}
extern "C" int kvsnap_pool_release(Pool* p, uint64_t id) {
    return guard([&] {
        std::lock_guard<std::mutex> lock(p->mutex);
        auto& a = allocation(p, id); check(cudaSetDevice(p->device));
        // Blocking release makes reuse safe even on a failing Python path.
        if (a.submitted) check(a.event_valid ? cudaEventSynchronize(a.event)
                                             : cudaStreamSynchronize(p->stream));
        check(cudaEventDestroy(a.event)); p->live.erase(id);
    });
}
