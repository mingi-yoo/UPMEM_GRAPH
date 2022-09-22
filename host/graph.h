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
    DPUGraph* dpu_param;
    uint32_t* row_ptr;
    uint32_t* col_idx;
    uint32_t* out_deg;
    float* value;
    float* output;
};

struct Graph_X {
    DPUGraph_X* dpu_param;
    uint32_t* row_ptr;
    uint32_t* col_idx;
    Feature* feature_c;
    Feature* feature_r;
    float* output;
};

static Graph read_csr(string csr_path) {
    Graph graph;
    graph.dpu_param = new DPUGraph[1];

    ifstream csr(csr_path);

    if (csr.is_open()) {
        csr >> graph.dpu_param[0].num_v >> graph.dpu_param[0].num_e;
        graph.dpu_param[0].num_v_origin = graph.dpu_param[0].num_v;
        graph.dpu_param[0].num_tiles = 1;

        uint32_t row_ptr_size = ROUND_UP_TO_MULTIPLE_OF_2(graph.dpu_param[0].num_v+1);
        uint32_t col_idx_size = ROUND_UP_TO_MULTIPLE_OF_2(graph.dpu_param[0].num_e);
        uint32_t feature_size = ROUND_UP_TO_MULTIPLE_OF_2(graph.dpu_param[0].num_v);

        graph.row_ptr = new uint32_t[row_ptr_size];
        graph.col_idx = new uint32_t[col_idx_size];
        graph.out_deg = new uint32_t[feature_size];
        graph.value = new float[feature_size];
        graph.output = new float[feature_size];

        for (uint32_t i = 0; i <= graph.dpu_param[0].num_v; i++) {
            csr >> graph.row_ptr[i];
        }

        for (uint32_t i = 0; i < graph.dpu_param[0].num_e; i++) {
            csr >> graph.col_idx[i];
        }

        for (uint32_t i = 0; i < graph.dpu_param[0].num_v; i++) {
            csr >> graph.out_deg[i];
            graph.value[i] = 1.0f / graph.dpu_param[0].num_v;
        };

        // set offset
        graph.dpu_param[0].row_ptr_start = ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph));
        graph.dpu_param[0].col_idx_start = graph.dpu_param[0].row_ptr_start + static_cast<unsigned>(row_ptr_size * sizeof(uint32_t));
        graph.dpu_param[0].value_start = graph.dpu_param[0].col_idx_start + static_cast<unsigned>(col_idx_size * sizeof(uint32_t));
        graph.dpu_param[0].out_deg_start = graph.dpu_param[0].value_start + static_cast<unsigned>(feature_size * sizeof(uint32_t));
        graph.dpu_param[0].output_start = graph.dpu_param[0].out_deg_start + static_cast<unsigned>(feature_size * sizeof(float));

        csr.close();

    }
    else
        throw invalid_argument("Cannot open graph");

    return graph;
}

void free_graph(Graph& graph) {
    delete [] graph.dpu_param;
    delete [] graph.row_ptr;
    delete [] graph.col_idx;
    delete [] graph.out_deg;
    delete [] graph.value;
    delete [] graph.output;
}

void free_graph(Graph_X& graph) {
    delete [] graph.dpu_param;
    delete [] graph.row_ptr;
    delete [] graph.col_idx;
    delete [] graph.feature_r;
    delete [] graph.feature_c;
    delete [] graph.output;
}

static Graph divide_graph_naive(Graph& graph, uint32_t n, uint32_t t) {
    Graph subgraph;

    subgraph.dpu_param = new DPUGraph[n];

    // distribute vertices in a balanced manner
    uint32_t num_v_origin = graph.dpu_param[0].num_v_origin;
    uint32_t q = num_v_origin / n;
    uint32_t r = num_v_origin - q * n;

    uint32_t q_t = num_v_origin / t;
    uint32_t q_r = num_v_origin - q_t * t;

    uint32_t *unit_v = new uint32_t[n];
    uint32_t *unit_t = new uint32_t[t];

    for (uint32_t i = 0; i < n; i++) {
        unit_v[i] = q;
    }
    for (uint32_t i = 0; i < r; i++) {
        unit_v[i]++;
    }

    // this is temporal (only used in this function)
    for (uint32_t i = 0; i < t; i++) {
        unit_t[i] = q_t;
    }
    for (uint32_t i = 0; i < q_r; i++) {
        unit_t[i]++;
    }

    cout<<"partitioned vertex check"<<endl;
    for (uint32_t i = 0; i < n; i++) {
        cout<<unit_v[i]<<" ";
    }
    cout<<endl;

    uint32_t col_idx_max = 0;
    uint32_t row_start = 0;
    uint32_t row_end = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t num_e;

        row_end += unit_v[i];
        num_e = graph.row_ptr[row_end] - graph.row_ptr[row_start];

        if (col_idx_max < num_e)
            col_idx_max = num_e;

        subgraph.dpu_param[i].num_v_origin = num_v_origin;
        subgraph.dpu_param[i].num_e = num_e;
        subgraph.dpu_param[i].num_tiles = t;

        row_start = row_end;
    }

    uint32_t row_ptr_size = ROUND_UP_TO_MULTIPLE_OF_2(unit_v[0]*t+1);
    uint32_t col_idx_size = ROUND_UP_TO_MULTIPLE_OF_2(col_idx_max);
    uint32_t feature_size = ROUND_UP_TO_MULTIPLE_OF_2(num_v_origin);
    uint32_t output_size = ROUND_UP_TO_MULTIPLE_OF_2(unit_v[0]);

    subgraph.row_ptr = new uint32_t[row_ptr_size * n];
    subgraph.col_idx = new uint32_t[col_idx_size * n];
    subgraph.out_deg = new uint32_t[feature_size];
    subgraph.value = new float[feature_size];
    subgraph.output = new float[output_size * n];

    for (uint32_t i = 0; i < num_v_origin; i++){
       subgraph.out_deg[i] = graph.out_deg[i];
       subgraph.value[i] = graph.value[i]; 
    }

    row_start = 0;
    row_end = 0;

    vector<vector<uint32_t>> vertices;
    vector<vector<uint32_t>> edges;
    vector<uint32_t> edge_acm(t);

    for (uint32_t i = 0; i < t; i++) {
        vertices.push_back(vector<uint32_t> ());
        edges.push_back(vector<uint32_t> ());
    }

    for (uint32_t i = 0; i < n; i++) {
        subgraph.dpu_param[i].num_v = unit_v[i];
        row_end += unit_v[i];

        subgraph.row_ptr[i*row_ptr_size] = 0;

        for (uint32_t j = row_start; j < row_end; j++) {
            for (uint32_t k = graph.row_ptr[j]; k < graph.row_ptr[j+1]; k++) {
                uint32_t col = graph.col_idx[k]; 
                for (uint32_t l = 0; l < t; l++) {
                    if (col >= unit_t[l]) {
                        egde_acm[l]++;
                        edges[l].push_back(col);
                        break;
                    }
                }
            }
            for (uint32_t k = 0; k < t; k++) {
                vertices[k].push_back(edge_acm[k]);
                edge_acm[k] = 0;
            }
        }

        subgraph.row_ptr[i*row_ptr_size] = 0;
        uint32_t idx = 1;
        uint32_t v_acm = 0;
        for (uint32_t j = 0; j < t; j++) {
            for (uint32_t k = 0; k < vertices[j].size(); k++) {
                v_acm += vertices[j][k]
                subgraph.row_ptr[i*row_ptr_size + idx] = v_acm;
                idx++;
            }
        }

        idx = 0;
        for (uint32_t j = 0; j < t; j++) {
            for (uint32_t k = 0; k < edges[j].size(); k++) {
                subgraph.col_idx[i*col_idx_size + idx] = edges[j][k];
                idx++;
            }
        }

        subgraph.dpu_param[i].row_ptr_start = ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph));
        subgraph.dpu_param[i].col_idx_start = subgraph.dpu_param[i].row_ptr_start + static_cast<unsigned>(row_ptr_size * sizeof(uint32_t));
        subgraph.dpu_param[i].value_start = subgraph.dpu_param[i].col_idx_start + static_cast<unsigned>(col_idx_size * sizeof(uint32_t));
        subgraph.dpu_param[i].out_deg_start = subgraph.dpu_param[i].value_start + static_cast<unsigned>(feature_size * sizeof(uint32_t));
        subgraph.dpu_param[i].output_start = subgraph.dpu_param[i].out_deg_start + static_cast<unsigned>(feature_size * sizeof(float));
         
        row_start = row_end;

        for (uint32_t j = 0; j < t; j++) {
            vertices[j].clear();
            edges[j].clear();
        }
    }

    delete [] unit_t;

    return subgraph;
}

static Graph_X divide_graph_ours(Graph& graph, uint32_t n) {
    Graph_X subgraph;

    subgraph.dpu_param = new DPUGraph_X;
    uint32_t num_v_origin = graph.dpu_param[0].num_v_origin;

    // make table for check
    vector<vector<bool>> e_check;
    for (uint32_t i = 0; i < n; i++)
        e_check.push_back(vector<bool> (num_v_origin));

    // calculate unit size
    uint32_t row_ptr_size = (graph.dpu_param[0].col_idx_start - graph.dpu_param[0].row_ptr_start) / sizeof(uint32_t);
    uint32_t col_idx_size = (graph.dpu_param[0].value_start - graph.dpu_param[0].col_idx_start) / sizeof(uint32_t);
    uint32_t output_size = ROUND_UP_TO_MULTIPLE_OF_2(graph.dpu_param[0].num_v);

    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = i*col_idx_size; j < i*col_idx_size + graph.dpu_param[i].num_e; j++)
            e_check[i][graph.col_idx[j]] = true;
    }

    vector<uint32_t> common_col;
    vector<vector<uint32_t>> respected_col;

    for (uint32_t i = 0; i < n; i++)
        respected_col.push_back(vector<uint32_t> ());

    for (uint32_t i = 0; i < num_v_origin; i++) {
        bool check = true;
        for (uint32_t j = 0; j < n; j++) {
            if (!e_check[j][i]) {
                check = false;
                break;
            }
        }
        if (check)
            common_col.push_back(i);
        else {
            for (uint32_t j = 0; j < n; j++) {
                if (e_check[j][i])
                    respected_col[j].push_back(i);
            }
        }
    }

    uint32_t feature_c_size = ROUND_UP_TO_MULTIPLE_OF_2(common_col.size());
    uint32_t feature_r_size = 0;

    for (uint32_t i = 0; i < n; i++) {
        if (respected_col[i].size() > feature_r_size)
            feature_r_size = respected_col[i].size();
    }
    feature_r_size = ROUND_UP_TO_MULTIPLE_OF_2(feature_r_size);

    // TO-DO: allocation features
    subgraph.row_ptr = new uint32_t[row_ptr_size * n];
    subgraph.col_idx = new uint32_t[col_idx_size * n];
    subgraph.feature_c = new Feature[feature_c_size];
    subgraph.feature_r = new Feature[feature_r_size * n];
    subgraph.output = new float[output_size * n];

    // copy data
    for (uint32_t i = 0; i < common_col.size(); i++) {
        uint32_t v_id = common_col[i];
        subgraph.feature_c[i].v_id = v_id;
        subgraph.feature_c[i].out_deg = graph.out_deg[vid];
        subgraph.feature_c[i].value = graph.value[vid];
    }

    for (uint32_t i = 0; i < n; i++) {
        subgraph.dpu_param[i].num_v_origin = num_v_origin;
        subgraph.dpu_param[i].num_v = graph.dpu_param[i].num_v;
        subgraph.dpu_param[i].num_e = graph.dpu_param[i].num_e;

        for (uint32_t j = 0; j <= graph.dpu_param[i].num_v; j++)
            subgraph.row_ptr[i*row_ptr_size+j] = graph.row_ptr[i*row_ptr_size+j];

        for (uint32_t j = 0; j < graph.dpu_param[i].num_e; j++)
            subgraph.col_idx[i*col_idx_size+j] = graph.col_idx[i*col_idx_size+j];

        for (uint32_t j = 0; j < respected_col[i].size(); j++) {
            uint32_t v_id = respected_col[i][j];
            uint32_t idx = i*feature_r_size+j;
            subgraph.feature_r[idx].v_id = v_id;
            subgraph.feature_r[idx].out_deg = graph.out_deg[v_id];
            subgraph.feature_r[idx].value = graph.value[v_id];
        }
    }
    // offset initialize
    for (uint32_t i = 0; i < n; i++) {
        subgraph.dpu_param[i].row_ptr_start = ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph_X));
        subgraph.dpu_param[i].col_idx_start = subgraph.dpu_param[i].row_ptr_start + static_cast<unsigned>(row_ptr_size * sizeof(uint32_t));
        subgraph.dpu_param[i].feature_c_start = subgraph.dpu_param[i].col_idx_start + static_cast<unsigned>(col_idx_size * sizeof(uint32_t));
        subgraph.dpu_param[i].feature_r_start = subgraph.dpu_param[i].feature_c_start + static_cast<unsigned>(feature_c_size * sizeof(Feature));
        subgraph.dpu_param[i].output_start = subgraph.dpu_param[i].feature_r_start + static_cast<unsigned>(feature_r_size * sizeof(Feature));
    }

    return subgraph;
}

#endif