#include <mram.h>
#include <mutex.h>
#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <float.h>

#include <alloc.h>
#include <barrier.h>
#include <defs.h>
#include <perfcounter.h>
#include <seqread.h>

#include "dpu_utils.h"
#include "../support/common.h"

BARRIER_INIT(my_barrier, NR_TASKLETS);
MUTEX_INIT(comp_mutex);

uint32_t change = 0;

int main() {
	if (me() == 0)
		mem_reset();
	barrier_wait(&my_barrier);

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

    uint32_t row_ptr_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->row_ptr_start;
	uint32_t col_idx_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->col_idx_start;
	uint32_t comp_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->comp_start;
	uint32_t flag_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->flag_start;

	uint64_t* cache_comp_u = mem_alloc(sizeof(uint64_t));
    uint64_t* cache_comp_v = mem_alloc(sizeof(uint64_t));
    uint64_t* cache_other = mem_alloc(sizeof(uint64_t));

	// TODO: making tasklet
	if (num_v_tasklet > 0) {
		uint32_t tasklet_row_ptr_m = row_ptr_m + row_start_tasklet * sizeof(uint32_t);
        seqreader_t row_ptr_reader;
        uint32_t* row_ptr = seqread_init(seqread_alloc(), (__mram_ptr void*)tasklet_row_ptr_m, &row_ptr_reader);
        uint32_t row_prev = *row_ptr;

        uint32_t col_start = *row_ptr;
        seqreader_t col_idx_reader;
        uint32_t* col_idx = seqread_init(seqread_alloc(), (__mram_ptr void*)(col_idx_m + col_start*sizeof(uint32_t)), &col_idx_reader);

        // TODO: making comp read
        mutex_id_t mutex_id = MUTEX_GET(comp_mutex);

        for (uint32_t i = 0; i < g_info->num_t; i++) {
        	uint32_t node_start_idx = g_info->node_start_idx + row_start_tasklet;
        	for (uint32_t j = 0; j < num_v_tasklet; j++) {
        		uint32_t comp_u = load4B(comp_m, node_start_idx, cache_comp_u);
        		row_ptr = seqread_get(row_ptr, sizeof(uint32_t), &row_ptr_reader);
        		uint32_t in_deg = *row_ptr - row_prev;

        		for (uint32_t k = 0; k < in_deg; k++) {
        			uint32_t comp_v = load4B(comp_m, *col_idx, cache_comp_v);
        			if (comp_u != comp_v) {
        				uint32_t high_comp = comp_u > comp_v ? comp_u : comp_v;
        				uint32_t low_comp = comp_u + (comp_v - high_comp);
        				uint32_t comp_high_comp = load4B(comp_m, high_comp, cache_other);
        				if (high_comp == comp_high_comp) {
        					mutex_lock(mutex_id);
        					change = 1;
        					store4B(low_comp, comp_m, high_comp, cache_other);
        					mutex_unlock(mutex_id);
        				}
        			}
        		}
        	}
        }
	}
	barrier_wait(&my_barrier);

	if (me() == 0)
		store4B(change, flag_m, 0, cache_other);
}