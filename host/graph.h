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
    uint32_t num_v_origin;
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
        graph.num_v_origin = graph.num_v;

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

static vector<Graph> divide_graph(Graph& graph, uint32_t n) {
    vector<Graph> subgraphs;

    uint32_t unit_v = ceil((float)graph.num_v/n);
    uint32_t last_v = graph.num_v - (n-1) * unit_v;

    for (uint32_t i = 0; i < n; i++) {
        Graph subgraph;
        subgraph.num_v_origin = graph.num_v;
        if (i != n-1) {
            subgraph.num_v = unit_v;
            subgraph.num_e = graph.row_ptr[(i+1)*unit_v] - graph.row_ptr[i*unit_v];

            subgraph.row_ptr.resize(ROUND_UP_TO_MULTIPLE_OF_2(subgraph.num_v+1));
            subgraph.col_idx.resize(ROUND_UP_TO_MULTIPLE_OF_2(subgraph.num_e));
            subgraph.out_deg.resize(ROUND_UP_TO_MULTIPLE_OF_2(graph.num_v));
            subgraph.value.resize(ROUND_UP_TO_MULTIPLE_OF_2(graph.num_v));

            subgraph.row_ptr[0] = 0;
            uint32_t bias = graph.row_ptr[i*unit_v];
            int idx = 1;

            for (uint32_t j = i*unit_v + 1; j <= (i+1)*unit_v; j++) {
                subgraph.row_ptr[idx] = graph.row_ptr[j] - bias;
                idx++;         
            }

            idx = 0;
            for (int j = graph.row_ptr[i*unit_v]; j < graph.row_ptr[(i+1)*unit_v]; j++) {
                subgraph.col_idx[idx] = graph.col_idx[j];
                idx++;
            }

            subgraph.out_deg = graph.out_deg;
            subgraph.value = graph.value;

        }
        else {
            subgraph.num_v = last_v;
            subgraph.num_e = graph.row_ptr[graph.num_v] - graph.row_ptr[i*unit_v];

            subgraph.row_ptr.resize(ROUND_UP_TO_MULTIPLE_OF_2(subgraph.num_v+1));
            subgraph.col_idx.resize(ROUND_UP_TO_MULTIPLE_OF_2(subgraph.num_e));
            subgraph.out_deg.resize(ROUND_UP_TO_MULTIPLE_OF_2(graph.num_v));
            subgraph.value.resize(ROUND_UP_TO_MULTIPLE_OF_2(graph.num_v));

            subgraph.row_ptr[0] = 0;
            uint32_t bias = graph.row_ptr[i*unit_v];
            int idx = 1;

            for (uint32_t j = i*unit_v + 1; j <= graph.num_v; j++){
                subgraph.row_ptr[idx] = graph.row_ptr[j] - bias;
                idx++;      
            } 
                   
            idx = 0;
            for (int j = graph.row_ptr[i*unit_v]; j < graph.row_ptr[graph.num_v]; j++) {
                subgraph.col_idx[idx] = graph.col_idx[j];
                idx++;
            }

            subgraph.out_deg = graph.out_deg;
            subgraph.value = graph.value;
        }
        subgraphs.push_back(subgraph);
    }
    return subgraphs;
}

#endif