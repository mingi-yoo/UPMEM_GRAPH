#ifndef _COMMON_H_
#define _COMMON_H_

#define ROUND_UP_TO_MULTIPLE_OF_2(x)    ((((x) + 1)/2)*2)
#define ROUND_UP_TO_MULTIPLE_OF_8(x)    ((((x) + 7)/8)*8)


struct DPUGraph {
    uint32_t num_v_origin;
    uint32_t num_v;
    uint32_t num_e;
    uint32_t num_t;
    uint32_t num_fc;
    uint32_t num_fr;

    uint32_t row_ptr_start;
    uint32_t col_idx_start;
    uint32_t fc_start;
    uint32_t fr_start;
    uint32_t output_start;
};

struct Feature {
    uint32_t out_deg;
    float value;
};

#endif