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
    vector<vector<DPUGraph>> dpu_param;
    vector<vector<uint32_t>> row_ptr;
    vector<vector<uint32_t>> col_idx;
    vector<vector<uint32_t>> out_deg;
    vector<vector<float>> value;
};

static Graph read_csr(string csr_path) {
    Graph graph;

    ifstream csr(csr_path);

    graph.dpu_param.resize(1);
    graph.row_ptr.resize(1);
    graph.col_idx.resize(1);
    graph.out_deg.resize(1);
    graph.value.resize(1);

    if (csr.is_open()) {
        graph.dpu_param[0].resize(1);
        csr >> graph.dpu_param[0][0].num_v >> graph.dpu_param[0][0].num_e;
        graph.dpu_param[0][0].num_v_origin = graph.dpu_param[0][0].num_v;

        graph.row_ptr[0].resize(ROUND_UP_TO_MULTIPLE_OF_2(graph.dpu_param[0][0].num_v+1));
        graph.col_idx[0].resize(ROUND_UP_TO_MULTIPLE_OF_2(graph.dpu_param[0][0].num_e));
        graph.out_deg[0].resize(ROUND_UP_TO_MULTIPLE_OF_2(graph.dpu_param[0][0].num_v));
        graph.value[0].resize(ROUND_UP_TO_MULTIPLE_OF_2(graph.dpu_param[0][0].num_v));

        for (uint32_t i = 0; i <= graph.dpu_param[0][0].num_v; i++) {
            uint32_t row;
            csr >> row;
            graph.row_ptr[0][i] = row;
        }

        for (uint32_t i = 0; i < graph.dpu_param[0][0].num_e; i++) {
            uint32_t col;
            csr >> col;
            graph.col_idx[0][i] = col;
        }

        for (uint32_t i = 0; i < graph.dpu_param[0][0].num_v; i++) {
            uint32_t deg;
            csr >> deg;
            graph.out_deg[0][i] = deg;
        }

        for (uint32_t i = 0; i < graph.dpu_param[0][0].num_v; i++)
            graph.value[0][i] = 1.0f / graph.dpu_param[0][0].num_v;

        csr.close();

    }
    else
        throw invalid_argument("Cannot open graph");

    // set offset
    graph.dpu_param[0][0].row_ptr_start = ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph));
    graph.dpu_param[0][0].col_idx_start = graph.dpu_param[0][0].row_ptr_start + static_cast<unsigned>(graph.row_ptr[0].size() * 4);
    graph.dpu_param[0][0].value_start = graph.dpu_param[0][0].col_idx_start + static_cast<unsigned>(graph.col_idx[0].size() * 4);
    graph.dpu_param[0][0].out_deg_start = graph.dpu_param[0][0].value_start + static_cast<unsigned>(graph.out_deg[0].size() * 4);
    graph.dpu_param[0][0].output_start = graph.dpu_param[0][0].out_deg_start + static_cast<unsigned>(graph.value[0].size() * 4);


    return graph;
}

static Graph divide_graph(Graph& graph, uint32_t n) {
    Graph subgraph;

    uint32_t unit_v = ceil((float)graph.dpu_param[0][0].num_v/n);
    uint32_t last_v = graph.dpu_param[0][0].num_v - (n-1) * unit_v;

    subgraph.dpu_param.resize(n);
    subgraph.row_ptr.resize(n);
    subgraph.col_idx.resize(n);
    subgraph.out_deg.resize(n);
    subgraph.value.resize(n);

    uint32_t row_ptr_max = 0;
    uint32_t col_idx_max = 0;

    for (uint32_t i = 0; i < n; i++) {
        subgraph.dpu_param[i].resize(1);
        subgraph.dpu_param[i][0].num_v_origin = graph.dpu_param[0][0].num_v;
        if (i != n-1) {
            subgraph.dpu_param[i][0].num_v = unit_v;
            subgraph.dpu_param[i][0].num_e = graph.row_ptr[0][(i+1)*unit_v] - graph.row_ptr[0][i*unit_v];

            subgraph.row_ptr[i].resize(ROUND_UP_TO_MULTIPLE_OF_2(subgraph.dpu_param[i][0].num_v+1));
            subgraph.col_idx[i].resize(ROUND_UP_TO_MULTIPLE_OF_2(subgraph.dpu_param[i][0].num_e));
            subgraph.out_deg[i].resize(ROUND_UP_TO_MULTIPLE_OF_2(graph.dpu_param[0][0].num_v));
            subgraph.value[i].resize(ROUND_UP_TO_MULTIPLE_OF_2(graph.dpu_param[0][0].num_v));

            subgraph.row_ptr[i][0] = 0;
            uint32_t bias = graph.row_ptr[0][i*unit_v];
            uint32_t idx = 1;

            for (uint32_t j = i*unit_v + 1; j <= (i+1)*unit_v; j++) {
                subgraph.row_ptr[i][idx] = graph.row_ptr[0][j] - bias;
                idx++;         
            }

            idx = 0;
            for (uint32_t j = graph.row_ptr[0][i*unit_v]; j < graph.row_ptr[0][(i+1)*unit_v]; j++) {
                subgraph.col_idx[i][idx] = graph.col_idx[0][j];
                idx++;
            }

            subgraph.out_deg[i] = graph.out_deg[0];
            subgraph.value[i] = graph.value[0];

        }
        else {
            subgraph.dpu_param[i][0].num_v = last_v;
            subgraph.dpu_param[i][0].num_e = graph.row_ptr[0][graph.dpu_param[0][0].num_v] - graph.row_ptr[0][i*unit_v];

            subgraph.row_ptr[i].resize(ROUND_UP_TO_MULTIPLE_OF_2(subgraph.dpu_param[i][0].num_v+1));
            subgraph.col_idx[i].resize(ROUND_UP_TO_MULTIPLE_OF_2(subgraph.dpu_param[i][0].num_e));
            subgraph.out_deg[i].resize(ROUND_UP_TO_MULTIPLE_OF_2(graph.dpu_param[0][0].num_v));
            subgraph.value[i].resize(ROUND_UP_TO_MULTIPLE_OF_2(graph.dpu_param[0][0].num_v));

            subgraph.row_ptr[i][0] = 0;
            uint32_t bias = graph.row_ptr[i][i*unit_v];
            uint32_t idx = 1;

            for (uint32_t j = i*unit_v + 1; j <= (i+1)*unit_v; j++) {
                subgraph.row_ptr[i][idx] = graph.row_ptr[0][j] - bias;
                idx++;         
            }

            idx = 0;
            for (uint32_t j = graph.row_ptr[0][i*unit_v]; j < graph.row_ptr[0][(i+1)*unit_v]; j++) {
                subgraph.col_idx[i][idx] = graph.col_idx[0][j];
                idx++;
            }

            subgraph.out_deg[i] = graph.out_deg[0];
            subgraph.value[i] = graph.value[0];
        }

        if (subgraph.row_ptr[i].size() > row_ptr_max)
            row_ptr_max = subgraph.row_ptr[i].size();
        if (subgraph.col_idx[i].size() > col_idx_max)
            col_idx_max = subgraph.col_idx[i].size();
    }

    // set offset

    for (uint32_t i = 0; i < n; i++) {
        graph.dpu_param[i][0].row_ptr_start = ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph));
        graph.dpu_param[i][0].col_idx_start = graph.dpu_param[i][0].row_ptr_start + static_cast<unsigned>(row_ptr_max * 4);
        graph.dpu_param[i][0].value_start = graph.dpu_param[i][0].col_idx_start + static_cast<unsigned>(col_idx_max * 4);
        graph.dpu_param[i][0].out_deg_start = graph.dpu_param[i][0].value_start + static_cast<unsigned>(graph.out_deg[0].size() * 4);
        graph.dpu_param[i][0].output_start = graph.dpu_param[i][0].out_deg_start + static_cast<unsigned>(graph.value[0].size() * 4);
    }

    return subgraph;
}

#endif