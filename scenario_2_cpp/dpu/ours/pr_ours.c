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

    printf("%d %d %d %d\n",g_info->num_v_origin, g_info->num_v, g_info->num_e, g_info->num_t);

    // initialize data offset
    uint32_t hash_fc_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->hash_fc_start;
    uint32_t hash_fr_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->hash_fr_start;
    uint32_t row_ptr_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->row_ptr_start;
    uint32_t col_idx_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->col_idx_start;
    uint32_t fc_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->fc_start;
    uint32_t fr_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->fr_start;
    uint32_t output_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->output_start;
    
    seqreader_t row_ptr_reader;
    uint32_t* row_ptr = seqread_init(seqread_alloc(), (__mram_ptr void*)row_ptr_m, &row_ptr_reader);

    uint32_t col_start = *row_ptr;
    seqreader_t col_idx_reader;
    uint32_t* col_idx = seqread_init(seqread_alloc(), (__mram_ptr void*)(col_idx_m + col_start*sizeof(uint32_t)), &col_idx_reader);

    uint32_t cache_size = 64;
    uint32_t output_cache_size = 64;

    uint32_t* hash_fc = mem_alloc(cache_size*sizeof(uint32_t));
    mram_read((__mram_ptr void const*)hash_fc_m, hash_fc, (cache_size+2)*sizeof(uint32_t));

    uint32_t* hash_fr = mem_alloc(cache_size*sizeof(uint32_t));
    mram_read((__mram_ptr void const*)hash_fr_m, hash_fr, (cache_size+2)*sizeof(uint32_t));

    struct Feature* fc = (struct Feature*) mem_alloc(cache_size*sizeof(struct Feature));
    mram_read((__mram_ptr void const*)fc_m, fc, cache_size*sizeof(struct Feature));    

    struct Feature* fr = (struct Feature*) mem_alloc(cache_size*sizeof(struct Feature));
    mram_read((__mram_ptr void const*)fr_m, fr, cache_size*sizeof(struct Feature));

    float* output = mem_alloc(output_cache_size*sizeof(float));

    // do pagerank    
    float kdamp = 0.85;

    uint32_t row_prev = *row_ptr;
    float base_score = 1.0f / g_info->num_v_origin;

    uint32_t cur_hash_fc_idx = 0;
    uint32_t cur_hash_fr_idx = 0;
    uint32_t cur_fc_idx = 0;
    uint32_t cur_fr_idx = 0;

    uint32_t num_v = g_info->num_v;
    uint32_t num_t = g_info->num_t;

    for (uint32_t i = 0; i < num_t; i++) {
        for (uint32_t j = 0; j < num_v; j++) {
            row_ptr = seqread_get(row_ptr, sizeof(uint32_t), &row_ptr_reader);
            uint32_t in_deg = *row_ptr - row_prev;
            float incoming_total = 0;
            float out_value = 0;

            for (uint32_t k = 0; k < in_deg; k++) {
                uint32_t col = *col_idx;
                // check hash value
                uint32_t hash_val = col % g_info->hash_key;

                // first_check: fc
                bool catch = false;
                uint32_t hash_idx = hash_val / cache_size;
                uint32_t hash_offset = hash_val % cache_size;
                if (cur_hash_fc_idx != hash_idx) {
                    mram_read((__mram_ptr void const*)(hash_fc_m+hash_idx*cache_size*sizeof(uint32_t)), hash_fc, (cache_size+2)*sizeof(uint32_t));
                    cur_hash_fc_idx = hash_idx;
                }
                uint32_t cache_idx, cache_offset;
                if (hash_fc[hash_offset] != hash_fc[hash_offset+1]) {
                    uint32_t need_to_check = hash_fc[hash_offset+1] - hash_fc[hash_offset];
                    uint32_t cur_check = 0;

                    cache_idx = hash_fc[hash_offset] / cache_size;
                    cache_offset = hash_fc[hash_offset] % cache_size;
                    if (cur_fc_idx != cache_idx) {
                        mram_read((__mram_ptr void const*)(fc_m+cache_idx*cache_size*sizeof(struct Feature)), fc, (cache_size)*sizeof(struct Feature));
                        cur_fc_idx = cache_idx;
                    }

                    while (cur_check < need_to_check) {
                        if (fc[cache_offset + cur_check].v_id == col) {
                            catch = true;
                            incoming_total += fc[cache_offset + cur_check].value / fc[cache_offset + cur_check].out_deg;
                            break;
                        }
                        cur_check++;
                        if (cache_offset + cur_check > cache_size) {
                            mram_read((__mram_ptr void const*)(fc_m+(cache_idx+1)*cache_size*sizeof(struct Feature)), fc, (cache_size)*sizeof(struct Feature));
                            cur_fc_idx = cache_idx + 1;
                            cache_offset = -cur_check;
                        }
                    }
                }
                // second_check: fr
                if (!catch) {
                     if (cur_hash_fr_idx != hash_idx) {
                        mram_read((__mram_ptr void const*)(hash_fr_m+hash_idx*cache_size*sizeof(uint32_t)), hash_fr, (cache_size+2)*sizeof(uint32_t));
                        cur_hash_fr_idx = hash_idx;
                    }
                    if (hash_fr[hash_offset] != hash_fr[hash_offset+1]) {
                        uint32_t need_to_check = hash_fr[hash_offset+1] - hash_fr[hash_offset];
                        uint32_t cur_check = 0;

                        cache_idx = hash_fr[hash_offset] / cache_size;
                        cache_offset = hash_fr[hash_offset] % cache_size;
                        if (cur_fr_idx != cache_idx) {
                            mram_read((__mram_ptr void const*)(fr_m+cache_idx*cache_size*sizeof(struct Feature)), fr, (cache_size)*sizeof(struct Feature));
                            cur_fr_idx = cache_idx;
                        }

                        while (cur_check < need_to_check) {
                            if (fr[cache_offset + cur_check].v_id == col) {
                                catch = true;
                                incoming_total += fr[cache_offset + cur_check].value / fr[cache_offset + cur_check].out_deg;
                                break;
                            }
                            cur_check++;
                            if (cache_offset + cur_check > cache_size) {
                                mram_read((__mram_ptr void const*)(fr_m+(cache_idx+1)*cache_size*sizeof(struct Feature)), fr, (cache_size)*sizeof(struct Feature));
                                cur_fr_idx = cache_idx + 1;
                                cache_offset = -cur_check;
                            }
                        }
                    } 
                }
                col_idx = seqread_get(col_idx, sizeof(uint32_t), &col_idx_reader);
            }
            row_prev = *row_ptr;
            out_value = kdamp * incoming_total;
            uint32_t output_idx = j / output_cache_size;
            uint32_t output_offset = j % output_cache_size;
            if (output_offset == 0 && i > 1)
                mram_read((__mram_ptr void const*)(output_m + output_idx*output_cache_size*sizeof(float)), output, output_cache_size*sizeof(float));

            if (i == 0)
                output[output_offset] = out_value;
            else
                output[output_offset] += out_value;

            if (i == num_t - 1)
                output[output_offset] += base_score;

            if (i == num_t - 1 && output_idx == 0 && output_offset < 10)
                printf("DPU RESULT: %f\n",output[output_offset]);

            if (output_offset == output_cache_size - 1 || j == num_v - 1)
                mram_write(output, (__mram_ptr void*)(output_m + output_idx*output_cache_size*sizeof(float)), output_cache_size*4);
        }
    }
    printf("DPU PAGERANK COMPLETE\n");

    return 0;
}
