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

int main() {
    uint32_t g_info_m = (uint32_t) DPU_MRAM_HEAP_POINTER;
    struct DPUGraph* g_info = (struct DPUGraph*) mem_alloc(ROUND_UP_TO_MULTIPLE_OF_8(sizeof(struct DPUGraph)));
    mram_read((__mram_ptr void const*)g_info_m, g_info, ROUND_UP_TO_MULTIPLE_OF_8(sizeof(struct DPUGraph)));

    printf("%d %d\n", param_w->num_v, param_w->num_e);

    uint32_t row_ptr_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->row_ptr_start;
    uint32_t col_idx_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->col_idx_start;
    uint32_t value_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->value_start;

    seqreader_t row_ptr_reader;
    uint32_t* row_ptr = seqread_init(seqread_alloc(), (__mram_ptr void*)row_ptr_m, &row_ptr_reader);

    seqreader_t col_idx_reader;
    uint32_t* col_idx = seqread_init(seqread_alloc(), (__mram_ptr void*)col_idx_m, &col_idx_reader);

    seqreader_t value_reader;
    float* value = seqread_init(seqread_alloc(), (__mram_ptr void*)value_m, &value_reader);

    for (int i = 0; i < 10; i++) {
        printf("%d %d %f\n", *row_ptr, *col_idx, *value);
        row_ptr = seqread_get(row_ptr, sizeof(uint32_t), &row_ptr_reader);
        col_idx = seqread_get(col_idx, sizeof(uint32_t), &col_idx_reader);
        value = seqread_get(value, sizeof(float), &value_reader);
    }

    return 0;
}
