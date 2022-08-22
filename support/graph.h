#ifndef _GRAPH_H_
#define _GRAPH_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"

using namespace std;

struct PRValue {
    uint32_t vid;
    float value;
};

struct Graph {
    uint32_t num_v;
    uint32_t num_e;
    vector<uint32_t> row_ptr;
    vector<uint32_t> col_idx;
    vector<PRValue> pr;
};

static Graph read_csr(string csr_path) {
    Graph graph;

    ifstream csr(csr_path);
    string line, temp;

    if (csr.is_open()) {
        csr >> graph.num_v >> graph.num_e;

        // parsing row_ptr
        getline(csr, line);
        stringstream ss(line);
        while (getline(ss, temp, ' '))
            graph.row_ptr.push_back(stoi(temp));

        // parsing col_idx
        ss.clear();
        getline(csr, line);
        ss.str(line);
        while (getline(ss, temp, ' ')) 
            graph.col_idx(stoi(temp));

        for (i = 0; i < graph.num_v; i++) {
            PRValue pr;
            pr.vid = i;
            pr.value = 1.0f / graph.num_v;
            graph.pr.push_back(pr);
        }

        csr.close();

    }
    else
        throw invalid_argument("Cannot open graph");

    return graph;
}