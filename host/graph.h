#ifndef _GRAPH_H_
#define _GRAPH_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "../support/common.h"

using namespace std;

struct Graph {
    uint32_t num_v;
    uint32_t num_e;
    vector<uint32_t> row_ptr;
    vector<uint32_t> col_idx;
    vector<uint32_t> out_deg;
    vector<float> value;
};

static Graph read_csr(string csr_path) {
    Graph graph;

    ifstream csr(csr_path);

    if (csr.is_open()) {
        csr >> graph.num_v >> graph.num_e;

        graph.row_ptr.resize(ROUND_UP_TO_MULTIPLE_OF_2(graph.num_v+1));
        graph.col_idx.resize(ROUND_UP_TO_MULTIPLE_OF_2(graph.num_e));
        graph.out_deg.resize(ROUND_UP_TO_MULTIPLE_OF_2(graph.num_v));
        graph.value.resize(ROUND_UP_TO_MULTIPLE_OF_2(graph.num_v));

        for (uint32_t i = 0; i <= graph.num_v; i++) {
            int row;
            csr >> row;
            graph.row_ptr[i] = row;
        }

        for (uint32_t i = 0; i < graph.num_e; i++) {
            int col;
            csr >> col;
            graph.col_idx[i] = col;
        }

        for (uint32_t i = 0; i < graph.num_v; i++) {
            int deg;
            csr >> deg;
            graph.out_deg[i] = deg;
        }

        for (uint32_t i = 0; i < graph.num_v; i++)
            graph.value[i] = 1.0f / graph.num_v;

        csr.close();

    }
    else
        throw invalid_argument("Cannot open graph");

    return graph;
}

#endif