#include <mram.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <float.h>

#define BUFFER_SIZE (1 << 16)

__mram_noinit uint32_t g_info[2];
__mram_noinit uint32_t row_ptr[BUFFER_SIZE];
__mram_noinit uint32_t col_idx[BUFFER_SIZE];
__mram_noinit float value[BUFFER_SIZE];

int main() {
    __dma_aligned uint32_t info[2];
    __dma_aligned uint32_t row[256];
    __dma_aligned uint32_t col[256];
    __dma_aligned float val[256];

    mram_read(g_info, info, sizeof(info));
    printf("%d, %d\n",info[0], info[1]);
    mram_read(row_ptr, row, sizeof(row));
    for (int i = 0; i < 10; i++)
        printf("%d\n", row[i]);
    mram_read(col_idx, col, sizeof(col));
    for (int i = 0; i < 10; i++)
        printf("%d\n", col[i]);
    mram_read(value, val, sizeof(val));
    for (int i = 0; i < 10; i++)
        printf("%f\n", val[i]);

    return 0;
}
