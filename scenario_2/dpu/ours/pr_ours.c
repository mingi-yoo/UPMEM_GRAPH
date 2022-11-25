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

BARRIER_INIT(my_barrier, NR_TASKLETS);

int main() {
    // reset MRAM heap
    if (me() == 0)
        mem_reset();

    barrier_wait(&my_barrier);

    // load graph data offset information
    uint32_t g_info_m = (uint32_t) DPU_MRAM_HEAP_POINTER;
    struct DPUGraph* g_info = (struct DPUGraph*) mem_alloc(ROUND_UP_TO_MULTIPLE_OF_8(sizeof(struct DPUGraph)));
    mram_read((__mram_ptr void const*)g_info_m, g_info, ROUND_UP_TO_MULTIPLE_OF_8(sizeof(struct DPUGraph)));

    uint32_t num_v_per_tasklet = ROUND_UP_TO_MULTIPLE_OF_2((g_info->num_v - 1) / NR_TASKLETS + 1);
    uint32_t row_start_tasklet = me() * num_v_per_tasklet;
    uint32_t num_v_tasklet;

    if (row_start_tasklet >= g_info->num_v)
        num_v_tasklet = 0;
    else if (row_start_tasklet + num_v_per_tasklet > g_info->num_v)
        num_v_tasklet = g_info->num_v - row_start_tasklet;
    else
        num_v_tasklet = num_v_per_tasklet;

     printf("%d: %d %d %d %d %d\n",me(), g_info->num_v_origin, g_info->num_v, g_info->num_e, g_info->num_t, num_v_tasklet);

    if (num_v_tasklet > 0) {
        // initialize data offset
        uint32_t row_ptr_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->row_ptr_start;
        uint32_t col_idx_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->col_idx_start;
        uint32_t fc_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->fc_start;
        uint32_t fr_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->fr_start;
        uint32_t output_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->output_start;
        
        uint32_t tasklet_row_ptr_m = row_ptr_m + row_start_tasklet * sizeof(uint32_t);
        seqreader_t row_ptr_reader;
        uint32_t* row_ptr = seqread_init(seqread_alloc(), (__mram_ptr void*)tasklet_row_ptr_m, &row_ptr_reader);

        uint32_t col_start = *row_ptr;
        seqreader_t col_idx_reader;
        uint32_t* col_idx = seqread_init(seqread_alloc(), (__mram_ptr void*)(col_idx_m + col_start*sizeof(uint32_t)), &col_idx_reader);

        uint32_t cache_size = 64;
        uint32_t output_cache_size = 2;

        struct Feature* fc = (struct Feature*) mem_alloc(cache_size*sizeof(struct Feature));
        mram_read((__mram_ptr void const*)fc_m, fc, cache_size*sizeof(struct Feature));    

        struct Feature* fr = (struct Feature*) mem_alloc(cache_size*sizeof(struct Feature));
        mram_read((__mram_ptr void const*)fr_m, fr, cache_size*sizeof(struct Feature));

        uint32_t tasklet_output_m = output_m + row_start_tasklet * sizeof(uint32_t);
        float* output = mem_alloc(output_cache_size*sizeof(float));

        // do pagerank    
        float kdamp = 0.85;

        uint32_t row_prev = *row_ptr;
        float base_score = (1.0f - kdamp) / g_info->num_v_origin;

        uint32_t cur_fc_idx = 0;
        uint32_t cur_fr_idx = 0;

        uint32_t num_v = num_v_tasklet;
        uint32_t num_t = g_info->num_t;

        for (uint32_t i = 0; i < num_t; i++) {
            for (uint32_t j = 0; j < num_v; j++) {
                row_ptr = seqread_get(row_ptr, sizeof(uint32_t), &row_ptr_reader);
                uint32_t in_deg = *row_ptr - row_prev;
                float incoming_total = 0;
                float out_value = 0;

                for (uint32_t k = 0; k < in_deg; k++) {
                    uint32_t col = *col_idx;
                    uint32_t cache_idx, cache_offset;
                    if (col < g_info->num_fc) {
                        cache_idx = col/cache_size;
                        cache_offset = col%cache_size;
                        if (cur_fc_idx != cache_idx) {
                            mram_read((__mram_ptr void const*)(fc_m+cache_idx*cache_size*sizeof(struct Feature)), fc, cache_size*sizeof(struct Feature));
                            cur_fc_idx = cache_idx;
                        }
                        incoming_total += fc[cache_offset].value / fc[cache_offset].out_deg;
                    }
                    else {
                        col -= g_info->num_fc;
                        cache_idx = col/cache_size;
                        cache_offset = col%cache_size;
                        if (cur_fr_idx != cache_idx) {
                            mram_read((__mram_ptr void const*)(fr_m+cache_idx*cache_size*sizeof(struct Feature)), fr, cache_size*sizeof(struct Feature));
                            cur_fr_idx = cache_idx;
                        }
                        incoming_total += fr[cache_offset].value / fr[cache_offset].out_deg;
                    }
                    col_idx = seqread_get(col_idx, sizeof(uint32_t), &col_idx_reader);
                }
                row_prev = *row_ptr;
                out_value = kdamp * incoming_total;
                uint32_t output_idx = j / output_cache_size;
                uint32_t output_offset = j % output_cache_size;
                if (output_offset == 0 && i >= 1)
                    mram_read((__mram_ptr void const*)(output_m + output_idx*output_cache_size*sizeof(float)), output, output_cache_size*sizeof(float));

                if (i == 0)
                    output[output_offset] = out_value;
                else
                    output[output_offset] += out_value;

                if (i == num_t - 1)
                    output[output_offset] += base_score;

                if (me() == 0 && i == num_t - 1 && output_idx == 0 && output_offset < 10)
                    printf("DPU RESULT: %f\n",output[output_offset]);

                if (output_offset == output_cache_size - 1 || j == num_v - 1)
                    mram_write(output, (__mram_ptr void*)(tasklet_output_m + output_idx*output_cache_size*sizeof(float)), output_cache_size*4);
            }
        }
    }
    
    
    printf("DPU PAGERANK COMPLETE\n");

    return 0;
}
