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

//BARRIER_INIT(my_barrier, NR_TASKLETS);

int main() {
    // reset MRAM heap
    //if (me() == 0)
    mem_reset();
   // barrier_wait(&my_barrier);

    // load graph data offset information
    uint32_t g_info_m = (uint32_t) DPU_MRAM_HEAP_POINTER;
    struct DPUGraph* g_info = (struct DPUGraph*) mem_alloc(ROUND_UP_TO_MULTIPLE_OF_8(sizeof(struct DPUGraph)));
    mram_read((__mram_ptr void const*)g_info_m, g_info, ROUND_UP_TO_MULTIPLE_OF_8(sizeof(struct DPUGraph)));

    printf("%d %d\n", g_info->num_v, g_info->num_e);

    // initialize data offset
    uint32_t row_ptr_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->row_ptr_start;
    uint32_t col_idx_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->col_idx_start;
    uint32_t value_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->value_start;
    uint32_t out_deg_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->out_deg_start;
    uint32_t output_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->output_start;

    seqreader_t row_ptr_reader;
    uint32_t* row_ptr = seqread_init(seqread_alloc(), (__mram_ptr void*)row_ptr_m, &row_ptr_reader);

    seqreader_t col_idx_reader;
    uint32_t* col_idx = seqread_init(seqread_alloc(), (__mram_ptr void*)col_idx_m, &col_idx_reader);

    uint32_t value_cache_size = 64;
    uint32_t out_deg_cache_size = 64;
    uint32_t output_cache_size = 64;

    float* value = mem_alloc(value_cache_size*sizeof(float));
    mram_read((__mram_ptr void const*)value_m, value, value_cache_size*4);

    uint32_t* out_deg = mem_alloc(out_deg_cache_size*sizeof(uint32_t));
    mram_read((__mram_ptr void const*)out_deg_m, out_deg, out_deg_cache_size*4);

    float* output = mem_alloc(output_cache_size*sizeof(float));

    // do pagerank    
    float kdamp = 0.85;

    uint32_t row_prev = *row_ptr;
    float base_score = 1.0f / g_info->num_v;

    uint32_t cur_cache_idx = 0;

    for (uint32_t i = 0; i < g_info->num_v; i++) {
        row_ptr = seqread_get(row_ptr, sizeof(uint32_t), &row_ptr_reader);
        uint32_t in_deg = *row_ptr - row_prev;
        float incoming_total = 0;
        float out_value = 0;

        for (uint32_t j = 0; j < in_deg; j++) {
            uint32_t col = *col_idx;
            uint32_t cache_idx = col/value_cache_size;
            uint32_t cache_offset = col%value_cache_size;
            if (cur_cache_idx != cache_idx) {
                mram_read((__mram_ptr void const*)(value_m+cache_idx*value_cache_size*sizeof(float)), value, value_cache_size*4);
                mram_read((__mram_ptr void const*)(out_deg_m+cache_idx*out_deg_cache_size*sizeof(uint32_t)), out_deg, out_deg_cache_size*4);
                cur_cache_idx = cache_idx;
            }
            incoming_total += value[cache_offset] / out_deg[cache_offset];
            col_idx = seqread_get(col_idx, sizeof(uint32_t), &col_idx_reader);
        }

	row_prev = *row_ptr;
        out_value = base_score + kdamp * incoming_total;
        uint32_t output_idx = i/output_cache_size;
        uint32_t output_offset = i%output_cache_size;
        output[output_offset] = out_value;
        if (output_idx == 0 && output_offset < 10)
            printf("DPU RESULT: %f\n",out_value);
        if (output_offset == output_cache_size - 1)
            mram_write(output, (__mram_ptr void*)(output_m + output_idx*output_cache_size*sizeof(float)), output_cache_size*4);
    }
    printf("DPU PAGERANK COMPLETE\n");

    return 0;
}
