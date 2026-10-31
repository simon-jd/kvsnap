// Independent syscall shim for the real C++ I/O implementation. Not a device.
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <chrono>
#include <thread>
#include <unistd.h>

extern "C" ssize_t __real_pwrite(int, const void*, size_t, off_t);
extern "C" ssize_t __real_pread(int, void*, size_t, off_t);
extern "C" int __real_fdatasync(int);

namespace {
struct Fault {
    std::atomic<int> interrupted{0}, error{0}, zero{0}, calls{0}, hold{0};
    std::atomic<size_t> limit{0};
};
Fault faults[3]; // pwrite, pread, fdatasync
bool before(int op, size_t* size) {
    auto& state = faults[op];
    state.calls++;
    while (state.hold.load()) std::this_thread::sleep_for(std::chrono::milliseconds(1));
    if (state.interrupted.load() > 0) {
        state.interrupted--;
        errno = EINTR;
        return false;
    }
    if (state.error.load()) {
        errno = state.error.load();
        return false;
    }
    if (size && state.limit.load() && *size > state.limit.load())
        *size = state.limit.load();
    return true;
}
}

extern "C" void kvsnap_test_io_reset() {
    for (auto& state : faults) {
        state.interrupted = 0; state.error = 0; state.zero = 0;
        state.calls = 0; state.limit = 0; state.hold = 0;
    }
}
extern "C" void kvsnap_test_io_fault(int op, int interrupted, size_t limit, int error, int zero) {
    faults[op].interrupted = interrupted;
    faults[op].limit = limit;
    faults[op].error = error;
    faults[op].zero = zero;
}
extern "C" int kvsnap_test_io_calls(int op) { return faults[op].calls.load(); }
extern "C" void kvsnap_test_io_hold(int op, int hold) { faults[op].hold = hold; }
extern "C" ssize_t __wrap_pwrite(int fd, const void* data, size_t size, off_t offset) {
    if (!before(0, &size)) return -1;
    if (faults[0].zero.load()) return 0;
    return __real_pwrite(fd, data, size, offset);
}
extern "C" ssize_t __wrap_pread(int fd, void* data, size_t size, off_t offset) {
    if (!before(1, &size)) return -1;
    if (faults[1].zero.load()) return 0;
    return __real_pread(fd, data, size, offset);
}
extern "C" int __wrap_fdatasync(int fd) {
    if (!before(2, nullptr)) return -1;
    return __real_fdatasync(fd);
}
