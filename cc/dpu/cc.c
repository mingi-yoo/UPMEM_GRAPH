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

#include "../../support/common.h"

BARRIER_INIT(my_barrier, NR_TASKLETS);
MUTEX_INIT(comp_mutex);

int main() {
	if (me() == 0)
		mem_reset();
	barrier_wait(&my_barrier);

	uint32_t g_info_m = (uint32_t) DPU_MRAM_HEAP_POINTER;
	struct DPUGraph* g_info = (sturct DPUGraph*) mem_alloc(ROUND_UP_TO_MULTIPLE_OF_8(sizeof(struct DPUGraph)));
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

	// TODO: making tasklet
	if (num_v_tasklet > 0) {
		uint32_t row_ptr_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->row_ptr_start;
		uint32_t col_idx_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->col_idx_start;
		uint32_t comp_m = (uint32_t)DPU_MRAM_HEAP_POINTER + g_info->comp_start;

		uint32_t tasklet_row_ptr_m = row_ptr_m + row_start_tasklet * sizeof(uint32_t);
        seqreader_t row_ptr_reader;
        uint32_t* row_ptr = seqread_init(seqread_alloc(), (__mram_ptr void*)tasklet_row_ptr_m, &row_ptr_reader);

        uint32_t col_start = *row_ptr;
        seqreader_t col_idx_reader;
        uint32_t* col_idx = seqread_init(seqread_alloc(), (__mram_ptr void*)(col_idx_m + col_start*sizeof(uint32_t)), &col_idx_reader);

        uint32_t cache_size = 64;

        uint32_t* comp_u = mem_alloc(8);
        uint32_t* comp_v = mem_alloc(cache_size*sizeof(uint32_t));

        uint32_t cur_comp_idx = 0;

        // TODO: making comp read
	}
}