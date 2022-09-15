#ifndef _GRAPH_H_
#define _GRAPH_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>

#include "../support/common.h"

using namespace std;

struct Graph {
    DPUGraph dpu_param;
    uint32_t* row_ptr;
    uint32_t* col_idx;
    uint32_t* out_deg;
    float* value;
    float* output;
};

static Graph read_csr(string csr_path) {
    Graph graph;

    ifstream csr(csr_path);

    if (csr.is_open()) {
        csr >> graph.dpu_param.num_v >> graph.dpu_param.num_e;
        graph.dpu_param.num_v_origin = graph.dpu_param.num_v;

        uint32_t row_ptr_size = ROUND_UP_TO_MULTIPLE_OF_2(graph.dpu_param.num_v+1);
        uint32_t col_idx_size = ROUND_UP_TO_MULTIPLE_OF_2(graph.dpu_param.num_e);
        uint32_t feature_size = ROUND_UP_TO_MULTIPLE_OF_2(graph.dpu_param.num_v);

        graph.row_ptr = new uint32_t[row_ptr_size];
        graph.col_idx = new uint32_t[col_idx_size];
        graph.out_deg = new uint32_t[feature_size];
        graph.value = new float[feature_size];
        graph.output = new float[feature_size];

        for (uint32_t i = 0; i <= graph.dpu_param.num_v; i++) {
            csr >> graph.row_ptr[i];
        }

        for (uint32_t i = 0; i < graph.dpu_param.num_e; i++) {
            csr >> graph.col_idx[i];
        }

        for (uint32_t i = 0; i < graph.dpu_param.num_v; i++) {
            csr >> graph.out_deg[i];
            graph.value[i] = 1.0f / graph.dpu_param.num_v;
        };

        // set offset
        graph.dpu_param.row_ptr_start = ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph));
        graph.dpu_param.col_idx_start = graph.dpu_param.row_ptr_start + static_cast<unsigned>(row_ptr_size * sizeof(uint32_t));
        graph.dpu_param.value_start = graph.dpu_param.col_idx_start + static_cast<unsigned>(col_idx_size * sizeof(uint32_t));
        graph.dpu_param.out_deg_start = graph.dpu_param.value_start + static_cast<unsigned>(feature_size * sizeof(uint32_t));
        graph.dpu_param.output_start = graph.dpu_param.out_deg_start + static_cast<unsigned>(feature_size * sizeof(float));

        csr.close();

    }
    else
        throw invalid_argument("Cannot open graph");

    return graph;
}

void free_graph(Graph& graph) {
    delete [] graph.row_ptr;
    delete [] graph.col_idx;
    delete [] graph.out_deg;
    delete [] graph.value;
    delete [] graph.output;
}

static Graph* divide_graph(Graph& graph, uint32_t n) {
    Graph* subgraph;

    subgraph = new Graph[n];

    uint32_t num_v_origin = graph.dpu_param.num_v_origin;
    uint32_t unit_v = ceil((float)num_v_origin / n);
    uint32_t last_v = num_v_origin - (n-1) * unit_v;

    uint32_t col_idx_max = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t num_e;
        if (i != n-1)
            num_e = graph.row_ptr[(i+1)*unit_v] - graph.row_ptr[i*unit_v];
        else
            num_e = graph.row_ptr[num_v_origin] - graph.row_ptr[i*unit_v];

        if (col_idx_max < num_e)
            col_idx_max = num_e;

        subgraph[i].dpu_param.num_v_origin = num_v_origin;
        subgraph[i].dpu_param.num_e = num_e;
    }

    uint32_t row_ptr_size = ROUND_UP_TO_MULTIPLE_OF_2(unit_v+1);
    uint32_t col_idx_size = ROUND_UP_TO_MULTIPLE_OF_2(col_idx_max);
    uint32_t feature_size = ROUND_UP_TO_MULTIPLE_OF_2(num_v_origin);

    for (uint32_t i = 0; i < n; i++) {
        if (i != n-1) {
            subgraph[i].dpu_param.num_v = unit_v;
            uint32_t output_size = ROUND_UP_TO_MULTIPLE_OF_2(unit_v);

            subgraph[i].row_ptr = new uint32_t[row_ptr_size];
            subgraph[i].col_idx = new uint32_t[col_idx_size];
            subgraph[i].out_deg = new uint32_t[feature_size];
            subgraph[i].value = new float[feature_size];
            subgraph[i].output = new float[output_size];

            subgraph[i].row_ptr[0] = 0;
            uint32_t bias = graph.row_ptr[i*unit_v];
            uint32_t idx = 1;

            for (uint32_t j = i*unit_v + 1; j <= (i+1)*unit_v; j++) {
                subgraph[i].row_ptr[idx] = graph.row_ptr[j] - bias;
                idx++;
            }

            idx = 0;
            for (uint32_t j = graph.row_ptr[i*unit_v]; j < graph.row_ptr[(i+1)*unit_v]; j++) {
                subgraph.col_idx[idx] = graph.col_idx[j];
                idx++;
            }
        }
        else {
            subgraph[i].dpu_param.num_v = last_v;
            uint32_t output_size = ROUND_UP_TO_MULTIPLE_OF_2(last_v);

            subgraph[i].row_ptr = new uint32_t[row_ptr_size];
            subgraph[i].col_idx = new uint32_t[col_idx_size];
            subgraph[i].out_deg = new uint32_t[feature_size];
            subgraph[i].value = new float[feature_size];
            subgraph[i].output = new float[output_size];

            subgraph[i].row_ptr[0] = 0;
            uint32_t bias = graph.row_ptr[i*unit_v];
            uint32_t idx = 1;

            for (uint32_t j = i*unit_v + 1; j <= num_v_origin; j++) {
                subgraph[i].row_ptr[idx] = graph.row_ptr[j] - bias;
                idx++;
            }

            idx = 0;
            for (uint32_t j = graph.row_ptr[i*unit_v]; j < graph.row_ptr[(i+1)*unit_v]; j++) {
                subgraph.col_idx[idx] = graph.col_idx[j];
                idx++;
            }
        }
    }
}

// static Graph divide_graph(Graph& graph, uint32_t n) {
//     Graph subgraph;

//     uint32_t num_v_origin = graph.dpu_param[0][0].num_v_origin;

//     uint32_t unit_v = ceil((float)num_v_origin/n);
//     uint32_t last_v = num_v_origin - (n-1) * unit_v;

//     // check max row & col size
//     uint32_t row_ptr_max = 0;
//     uint32_t col_idx_max = 0;

//     if (unit_v > last_v)
//         row_ptr_max = ROUND_UP_TO_MULTIPLE_OF_2(unit_v+1);
//     else
//         row_ptr_max = ROUND_UP_TO_MULTIPLE_OF_2(last_v+1);

//     for (uint32_t i = 0; i < n; i++) {
//         uint32_t num_e;
//         if (i != n-1)
//             num_e = graph.row_ptr[0][(i+1)*unit_v] - graph.row_ptr[0][i*unit_v];
//         else
//             num_e = graph.row_ptr[0][num_v_origin] -  graph.row_ptr[0][i*unit_v];

//         if (col_idx_max < num_e)
//             col_idx_max = num_e;
//     }

//     col_idx_max = ROUND_UP_TO_MULTIPLE_OF_2(col_idx_max);

//     for (uint32_t i = 0; i < n; i++) {
//         vector<DPUGraph> dpu_param;
//         vector<uint32_t> row_ptr;
//         vector<uint32_t> col_idx;
//         vector<uint32_t> out_deg;
//         vector<float> value;

//         DPUGraph dpu_param_temp;

//         dpu_param_temp.num_v_origin = num_v_origin;
//         if (i != n-1) {
//             dpu_param_temp.num_v = unit_v;
//             dpu_param_temp.num_e = graph.row_ptr[0][(i+1)*unit_v] - graph.row_ptr[0][i*unit_v];

//             row_ptr.push_back(0);
//             uint32_t bias = graph.row_ptr[0][i*unit_v];
//             uint32_t idx = 1;

//             for (uint32_t j = i*unit_v + 1; j <= (i+1)*unit_v; j++) {
//                 row_ptr.push_back(graph.row_ptr[0][j] - bias);
//                 idx++;         
//             }

//             idx = 0;
//             for (uint32_t j = graph.row_ptr[0][i*unit_v]; j < graph.row_ptr[0][(i+1)*unit_v]; j++) {
//                 col_idx.push_back(graph.col_idx[0][j]);
//                 idx++;
//             }

//             out_deg = graph.out_deg[0];
//             value = graph.value[0];

//         }
//         else {
//             dpu_param_temp.num_v = last_v;
//             dpu_param_temp.num_e = graph.row_ptr[0][num_v_origin] - graph.row_ptr[0][i*unit_v];

//             row_ptr.push_back(0);
//             uint32_t bias = graph.row_ptr[0][i*unit_v];
//             uint32_t idx = 1;

//             for (uint32_t j = i*unit_v + 1; j <= last_v; j++) {
//                 row_ptr.push_back(graph.row_ptr[0][j] - bias);
//                 idx++;         
//             }

//             idx = 0;
//             for (uint32_t j = graph.row_ptr[0][i*unit_v]; j < graph.row_ptr[0][num_v_origin]; j++) {
//                 col_idx.push_back(graph.col_idx[0][j]);
//                 idx++;
//             }

//             out_deg = graph.out_deg[0];
//             value = graph.value[0];
//         }
//         dpu_param.push_back(dpu_param_temp);
//         row_ptr.resize(row_ptr_max);
//         col_idx.resize(col_idx_max);

//         // set offset
//         dpu_param[0].row_ptr_start = ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph));
//         dpu_param[0].col_idx_start = dpu_param[0].row_ptr_start + static_cast<unsigned>(row_ptr_max * 4);
//         dpu_param[0].value_start = dpu_param[0].col_idx_start + static_cast<unsigned>(col_idx_max * 4);
//         dpu_param[0].out_deg_start = dpu_param[0].value_start + static_cast<unsigned>(out_deg.size() * 4);
//         dpu_param[0].output_start = dpu_param[0].out_deg_start + static_cast<unsigned>(value.size() * 4);

//         subgraph.dpu_param.push_back(dpu_param);
//         subgraph.row_ptr.push_back(row_ptr);
//         subgraph.col_idx.push_back(col_idx);
//         subgraph.value.push_back(value);
//         subgraph.out_deg.push_back(out_deg);
//     }

//     return subgraph;
// }

#endif