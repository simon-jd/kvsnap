#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <unistd.h>
extern "C" uint32_t kvsnap_crc32c(const void*, size_t);
extern "C" int kvsnap_pwrite_all(int, const void*, size_t, int64_t, int);
int main() {
    assert(kvsnap_crc32c("123456789", 9) == 0xe3069283U);
    assert(kvsnap_crc32c("", 0) == 0);
    FILE* f = tmpfile(); assert(f);
    assert(kvsnap_pwrite_all(fileno(f), "kv-bytes", 8, 3, 1) == 0);
    char out[8]; assert(pread(fileno(f), out, 8, 3) == 8);
    assert(memcmp(out, "kv-bytes", 8) == 0);
    assert(kvsnap_pwrite_all(-1, "x", 1, 0, 0) != 0);
    assert(kvsnap_pwrite_all(fileno(f), "x", 1, -1, 0) != 0);
    fclose(f);
}
