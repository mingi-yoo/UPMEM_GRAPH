#include <mram.h>
#include <stdint.h>
#include <stdbool.h>
#include <float.h>

#define BUFFER_SIZE (1 << 16)

__mram_noinit uint32_t row_ptr[BUFER_SIZE];
__mram_noinit uint32_t col_idx[BUFFER_SIZE];
__mram_noinit float value[BUFFER_SIZE];

int main() {
    __dma_aligned uint32_t row[256];
    __dma_aligned uint32_t col[256];
    __dma_aligned float val[256];

    mram_read(row_ptr, row, sizeof(row));

    return 0;
}
