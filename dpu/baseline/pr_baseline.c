#include <mram.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <float.h>

#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <mram.h>
#include <perfcounter.h>
#include <seqread.h>

#include "../../support/common.h"

#define BUFFER_SIZE (1 << 16)

// __mram_noinit uint32_t g_info_m[2];
// __mram_noinit uint32_t row_ptr_m[BUFFER_SIZE];
// __mram_noinit uint32_t col_idx_m[BUFFER_SIZE];
// __mram_noinit float value_m[BUFFER_SIZE];

int main() {
    // __dma_aligned uint32_t g_info[2];
    // __dma_aligned uint32_t row_ptr[BUFFER_SIZE];
    // __dma_aligned uint32_t col_idx[BUFFER_SIZE];
    // __dma_aligned float value[BUFFER_SIZE];

    uint32_t param_m = (uint32_t) DPU_MRAM_HEAP_POINTER;
    struct DPUGraph* param_w = (struct DPUGraph*) mem_alloc(ROUND_UP_TO_MULTIPLE_OF_8(sizeof(struct DPUGraph)));
    mram_read((__mram_ptr void const*)param_m, param_w, ROUND_UP_TO_MULTIPLE_OF_8(sizeof(struct DPUGraph)));

    printf("%d %d\n", param_w->num_v, param_w->num_e);
    // // read data to wram
    // mram_read(g_info_m, g_info, sizeof(g_info));
    // mram_read(row_ptr_m, row_ptr, sizeof(row_ptr));
    // mram_read(col_idx_m, col_idx, sizeof(col_idx));
    // mram_read(value_m, value, sizeof(value));

    return 0;
}
