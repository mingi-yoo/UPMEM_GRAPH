#ifndef _GRAPH_H_
#define _GRAPH_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include "common.h"

using namespace std;

struct PRValue {
    uint32_t vid;
    float value;
};

struct Graph {
    uint32_t num_v;
    uint32_t num_e;
    uint32_t* row_ptr;
    uint32_t* col_idx;
    PRValue* pr;
};

static Graph read_csr(string csr_path) {
    Graph graph;

    ifstream csr(csr_path);
    string line, temp;

    if (csr.is_open()) {
        csr >> graph.num_v >> graph.num_e;

        int v_rnd = ROUND_UP_TO_MULTIPLE_OF_8(graph.num_v+1);
        int e_rnd = ROUND_UP_TO_MULTIPLE_OF_8(graph.num_e);

        graph.row_ptr = new uint32_t[v_rnd];
        graph.col_idx = new uint32_t[e_rnd];
        graph.pr = new PRValue[v_rnd];

        int i = 0;

        // parsing row_ptr
        getline(csr, line);
        stringstream ss(line);
        while (getline(ss, temp, ' ')) {
            graph.row_ptr[i] = stoi(temp);
            i++;
        }

        // parsing col_idx
        i = 0;
        ss.clear();
        getline(csr, line);
        ss.str(line);
        while (getline(ss, temp, ' ')) {
            graph.col_idx[i] = stoi(temp);
            i++;
        }

        for (i = 0; i < graph.num_v; i++) {
            graph.pr[i].vid = i;
            graph.pr[i].value = 1.0f / graph.num_v;
        }

        csr.close();

    }
    else
        throw invalid_argument("Cannot open graph");

    return graph;
}

static void free_graph (Graph* graph) {
    delete [] graph->row_ptr;
    delete [] graph->col_idx;
    delete [] graph->pr;
}