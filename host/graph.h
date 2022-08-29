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

    vector<DPUGraph> dpu_param;
    vector<uint32_t> row_ptr;
    vector<uint32_t> col_idx;
    vector<uint32_t> out_deg;
    vector<float> value;

    if (csr.is_open()) {
        DPUGraph dpu_param_temp;

        csr >> dpu_param_temp.num_v >> dpu_param_temp.num_e;
        dpu_param_temp.num_v_origin = dpu_param_temp.num_v;

        for (uint32_t i = 0; i <= dpu_param_temp.num_v; i++) {
            uint32_t row;
            csr >> row;
            row_ptr.push_back(row);
        }

        for (uint32_t i = 0; i < dpu_param_temp.num_e; i++) {
            uint32_t col;
            csr >> col;
            col_idx.push_back(col);
        }

        for (uint32_t i = 0; i < dpu_param_temp.num_v; i++) {
            uint32_t deg;
            csr >> deg;
            out_deg.push_back(deg);
        }

        for (uint32_t i = 0; i < dpu_param_temp.num_v; i++)
            value.push_back(1.0f / dpu_param_temp.num_v);

        csr.close();

        dpu_param.push_back(dpu_param_temp);

    }
    else
        throw invalid_argument("Cannot open graph");

    row_ptr.resize(ROUND_UP_TO_MULTIPLE_OF_2(dpu_param[0].num_v+1));
    col_idx.resize(ROUND_UP_TO_MULTIPLE_OF_2(dpu_param[0].num_e));
    out_deg.resize(ROUND_UP_TO_MULTIPLE_OF_2(dpu_param[0].num_v));
    value.resize(ROUND_UP_TO_MULTIPLE_OF_2(dpu_param[0].num_v));

    // set offset
    dpu_param[0].row_ptr_start = ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph));
    dpu_param[0].col_idx_start = dpu_param[0].row_ptr_start + static_cast<unsigned>(row_ptr.size() * 4);
    dpu_param[0].value_start = dpu_param[0].col_idx_start + static_cast<unsigned>(col_idx.size() * 4);
    dpu_param[0].out_deg_start = dpu_param[0].value_start + static_cast<unsigned>(out_deg.size() * 4);
    dpu_param[0].output_start = dpu_param[0].out_deg_start + static_cast<unsigned>(value.size() * 4);

    graph.dpu_param.push_back(dpu_param);
    graph.row_ptr.push_back(row_ptr);
    graph.col_idx.push_back(col_idx);
    graph.value.push_back(value);
    graph.out_deg.push_back(out_deg);

    return graph;
}

static Graph divide_graph(Graph& graph, uint32_t n) {
    Graph subgraph;

    uint32_t unit_v = ceil((float)graph.dpu_param[0][0].num_v/n);
    uint32_t last_v = graph.dpu_param[0][0].num_v - (n-1) * unit_v;

    // check max row & col size
    uint32_t row_ptr_max = 0;
    uint32_t col_idx_max = 0;

    if (unit_v > last_v)
        row_ptr_max = ROUND_UP_TO_MULTIPLE_OF_2(unit_v+1);
    else
        row_ptr_max = ROUND_UP_TO_MULTIPLE_OF_2(last_v+1);

    for (uint32_t i = 0; i < n; i++) {
        uint32_t num_e;
        if (i != n-1)
            num_e = graph.row_ptr[0][(i+1)*unit_v] - graph.row_ptr[0][i*unit_v];
        else
            num_e = graph.row_ptr[0].back() -  graph.row_ptr[0][i*unit_v];

        if (col_idx_max < num_e)
            col_idx_max = num_e;
    }

    col_idx_max = ROUND_UP_TO_MULTIPLE_OF_2(col_idx_max);

    for (uint32_t i = 0; i < n; i++) {
        vector<DPUGraph> dpu_param;
        vector<uint32_t> row_ptr;
        vector<uint32_t> col_idx;
        vector<uint32_t> out_deg;
        vector<float> value;

        DPUGraph dpu_param_temp;

        dpu_param_temp.num_v_origin = graph.dpu_param[0][0].num_v;
        if (i != n-1) {
            dpu_param_temp.num_v = unit_v;
            dpu_param_temp.num_e = graph.row_ptr[0][(i+1)*unit_v] - graph.row_ptr[0][i*unit_v];

            row_ptr.push_back(0);
            uint32_t bias = graph.row_ptr[0][i*unit_v];
            uint32_t idx = 1;

            for (uint32_t j = i*unit_v + 1; j <= (i+1)*unit_v; j++) {
                row_ptr.push_back(graph.row_ptr[0][j] - bias);
                idx++;         
            }

            idx = 0;
            for (uint32_t j = graph.row_ptr[0][i*unit_v]; j < graph.row_ptr[0][(i+1)*unit_v]; j++) {
                col_idx.push_back(graph.col_idx[0][j]);
                idx++;
            }

            out_deg = graph.out_deg[0];
            value = graph.value[0];

        }
        else {
            dpu_param_temp.num_v = last_v;
            dpu_param_temp.num_e = graph.row_ptr[0].back() - graph.row_ptr[0][i*unit_v];

            row_ptr.push_back(0);
            uint32_t bias = graph.row_ptr[0][i*unit_v];
            uint32_t idx = 1;

            for (uint32_t j = i*unit_v + 1; j <= last_v; j++) {
                subgraph.row_ptr[i][idx] = graph.row_ptr[0][j] - bias;
                idx++;         
            }

            idx = 0;
            for (uint32_t j = graph.row_ptr[0][i*unit_v]; j < graph.row_ptr[0].back(); j++) {
                subgraph.col_idx[i][idx] = graph.col_idx[0][j];
                idx++;
            }

            out_deg = graph.out_deg[0];
            value = graph.value[0];
        }
        dpu_param.push_back(dpu_param_temp);
        row_ptr.resize(row_ptr_max);
        col_idx.resize(col_idx_max);

        // set offset
        dpu_param[0].row_ptr_start = ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph));
        dpu_param[0].col_idx_start = dpu_param[0].row_ptr_start + static_cast<unsigned>(row_ptr_max * 4);
        dpu_param[0].value_start = dpu_param[0].col_idx_start + static_cast<unsigned>(col_idx_max * 4);
        dpu_param[0].out_deg_start = dpu_param[0].value_start + static_cast<unsigned>(out_deg.size() * 4);
        dpu_param[0].output_start = dpu_param[0].out_deg_start + static_cast<unsigned>(value.size() * 4);

        subgraph.dpu_param.push_back(dpu_param);
        subgraph.row_ptr.push_back(row_ptr);
        subgraph.col_idx.push_back(col_idx);
        subgraph.value.push_back(value);
        subgraph.out_deg.push_back(out_deg);
    }

    return subgraph;
}

#endif