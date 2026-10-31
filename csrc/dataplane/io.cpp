#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <deque>
#include <future>
#include <mutex>
#include <thread>
#include <vector>
#include <unistd.h>
#if defined(__x86_64__)
#include <nmmintrin.h>
#include <cstring>
#endif

static uint32_t portable_crc(const void* data, size_t size, uint32_t prior) {
    auto bytes = static_cast<const uint8_t*>(data);
    uint32_t crc = ~prior;
    for (size_t i = 0; i < size; ++i) {
        crc ^= bytes[i];
        for (int b = 0; b < 8; ++b)
            crc = (crc >> 1) ^ (0x82f63b78U & (0U - (crc & 1U)));
    }
    return ~crc;
}
#if defined(__x86_64__)
__attribute__((target("sse4.2")))
static uint32_t accelerated_crc(const void* data, size_t size, uint32_t prior) {
    auto bytes = static_cast<const uint8_t*>(data);
    uint64_t crc = ~prior;
    while (size >= 8) {
        uint64_t word; std::memcpy(&word, bytes, 8);
        crc = _mm_crc32_u64(crc, word); bytes += 8; size -= 8;
    }
    while (size--) crc = _mm_crc32_u8(static_cast<uint32_t>(crc), *bytes++);
    return ~static_cast<uint32_t>(crc);
}
#endif
extern "C" uint32_t kvsnap_crc32c_extend(const void* data, size_t size, uint32_t prior) {
#if defined(__x86_64__)
    if (__builtin_cpu_supports("sse4.2")) return accelerated_crc(data, size, prior);
#endif
    return portable_crc(data, size, prior);
}
extern "C" uint32_t kvsnap_crc32c(const void* data, size_t size) {
    return kvsnap_crc32c_extend(data, size, 0);
}
extern "C" int kvsnap_io_abi_version() { return 1; }

// The caller owns fd and directory commit ordering. Returns errno, not -1.
extern "C" int kvsnap_pwrite_all(int fd, const void* data, size_t size,
                               int64_t offset, int durable) {
    if (offset < 0 || size > static_cast<uint64_t>(INT64_MAX - offset))
        return EINVAL;
    size_t written = 0;
    while (written < size) {
        const auto count = pwrite(fd, static_cast<const char*>(data) + written,
                                  size - written, offset + written);
        if (count < 0) { if (errno == EINTR) continue; return errno; }
        if (count == 0) return EIO;
        written += static_cast<size_t>(count);
    }
    if (durable) {
        int rc;
        do { rc = fdatasync(fd); } while (rc < 0 && errno == EINTR);
        if (rc < 0) return errno;
    }
    return 0;
}

extern "C" int kvsnap_pread_all(int fd, void* data, size_t size, int64_t offset) {
    if (offset < 0 || size > static_cast<uint64_t>(INT64_MAX - offset)) return EINVAL;
    size_t done = 0;
    while (done < size) {
        const auto count = pread(fd, static_cast<char*>(data) + done, size - done, offset + done);
        if (count < 0) { if (errno == EINTR) continue; return errno; }
        if (count == 0) return EIO;
        done += static_cast<size_t>(count);
    }
    return 0;
}

namespace {
struct Job {
    int fd, write, durable;
    void* data;
    size_t size;
    int64_t offset;
    std::promise<int> result;
};
struct IOPool {
    size_t limit, outstanding{0};
    bool stopping{false};
    std::mutex lock;
    std::condition_variable work, capacity;
    std::deque<std::shared_ptr<Job>> queue;
    std::vector<std::thread> workers;
    IOPool(size_t count, size_t max_bytes) : limit(max_bytes) {
        try {
            for (size_t i = 0; i < count; ++i) workers.emplace_back([this] { run(); });
        } catch (...) { stop(); throw; }
    }
    void stop() {
        { std::lock_guard<std::mutex> guard(lock); stopping = true; }
        work.notify_all(); capacity.notify_all();
        for (auto& worker : workers) if (worker.joinable()) worker.join();
    }
    ~IOPool() { stop(); }
    void run() {
        for (;;) {
            std::shared_ptr<Job> job;
            {
                std::unique_lock<std::mutex> guard(lock);
                work.wait(guard, [&] { return stopping || !queue.empty(); });
                if (queue.empty()) return;
                job = queue.front(); queue.pop_front();
            }
            const int result = job->write
                ? kvsnap_pwrite_all(job->fd, job->data, job->size, job->offset, job->durable)
                : kvsnap_pread_all(job->fd, job->data, job->size, job->offset);
            { std::lock_guard<std::mutex> guard(lock); outstanding -= job->size; }
            capacity.notify_all();
            job->result.set_value(result);
        }
    }
};
}
// Synchronous ABI over a bounded C++ worker pool. Buffers and fd are borrowed
// until return; the ctypes caller retains them while the GIL is released.
extern "C" void* kvsnap_io_pool_create(size_t workers, size_t max_bytes) {
    if (!workers || workers > 64 || !max_bytes) return nullptr;
    try { return new IOPool(workers, max_bytes); } catch (...) { return nullptr; }
}
extern "C" void kvsnap_io_pool_destroy(IOPool* pool) { delete pool; }
extern "C" int kvsnap_io_pool_transfer(IOPool* pool, int fd, void* data, size_t size,
                                     int64_t offset, int write, int durable) {
    if (!pool || (!data && size) || (write != 0 && write != 1)) return EINVAL;
    if (size > pool->limit) return EMSGSIZE;
    try {
        auto job = std::make_shared<Job>();
        job->fd = fd; job->data = data; job->size = size; job->offset = offset;
        job->write = write; job->durable = durable;
        auto result = job->result.get_future();
        {
            std::unique_lock<std::mutex> guard(pool->lock);
            pool->capacity.wait(guard, [&] { return pool->stopping || size <= pool->limit - pool->outstanding; });
            if (pool->stopping) return ECANCELED;
            pool->queue.push_back(job);
            pool->outstanding += size;
        }
        pool->work.notify_one();
        return result.get();
    } catch (...) { return ENOMEM; }
}
