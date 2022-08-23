#ifndef _GRAPH_H_
#define _GRAPH_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#define ROUND_UP_TO_MULTIPLE_OF_2(x)    ((((x) + 1)/2)*2)
#define ROUND_UP_TO_MULTIPLE_OF_8(x)    ((((x) + 7)/8)*8)

using namespace std;

struct Graph {
    uint32_t num_v;
    uint32_t num_e;
    vector<uint32_t> row_ptr;
    vector<uint32_t> col_idx;
    vector<float> value;
};

static Graph read_csr(string csr_path) {
    Graph graph;

    ifstream csr(csr_path);

    if (csr.is_open()) {
        csr >> graph.num_v >> graph.num_e;

        graph.row_ptr.resize(ROUND_UP_TO_MULTIPLE_OF_2(graph.num_v+1));
        graph.col_idx.resize(ROUND_UP_TO_MULTIPLE_OF_2(graph.num_e));
        graph.value.resize(ROUND_UP_TO_MULTIPLE_OF_2(graph.num_v));

        for (int i = 0; i <= graph.num_v; i++) {
            int row;
            csr >> row;
            graph.row_ptr[i] = row;
        }

        for (int i = 0; i < graph.num_e; i++) {
            int col;
            csr >> col;
            graph.col_idx[i] = col;
        }

        for (int i = 0; i < graph.num_v; i++)
            graph.value[i] = 1.0f / graph.num_v;

        csr.close();

    }
    else
        throw invalid_argument("Cannot open graph");

    return graph;
}

#endif