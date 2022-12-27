#ifndef _GRAPH_H_
#define _GRAPH_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <iterator>
#include <algorithm>

#include "../support/common.h"

using namespace std;

struct Graph {
    vector<vector<DPUGraph>> dpu_param;
    vector<vector<uint32_t>> row_ptr;
    vector<vector<uint32_t>> col_idx;
    vector<uint32_t> comp;
};

static Graph read_csr(string csr_path) {
    Graph graph;

    ifstream csr(csr_path);

    if (csr.is_open()) {
        graph.dpu_param.push_back(vector<DPUGraph> (1));

        csr >> graph.dpu_param[0][0].num_v >> graph.dpu_param[0][0].num_e;
        graph.dpu_param[0][0].num_v_origin = graph.dpu_param[0][0].num_v;
        graph.dpu_param[0][0].num_t = 1;

        uint32_t row_ptr_size = ROUND_UP_TO_MULTIPLE_OF_2(graph.dpu_param[0][0].num_v+1);
        uint32_t col_idx_size = ROUND_UP_TO_MULTIPLE_OF_2(graph.dpu_param[0][0].num_e);
        uint32_t comp_size = ROUND_UP_TO_MULTIPLE_OF_2(graph.dpu_param[0][0].num_v);

        graph.row_ptr.push_back(vector<uint32_t> (row_ptr_size));
        graph.col_idx.push_back(vector<uint32_t> (col_idx_size));
        graph.comp.resize(comp_size);

        for (uint32_t i = 0; i <= graph.dpu_param[0][0].num_v; i++) {
            csr >> graph.row_ptr[0][i];
        }

        for (uint32_t i = 0; i < graph.dpu_param[0][0].num_e; i++) {
            csr >> graph.col_idx[0][i];
        }

        // set offset
        graph.dpu_param[0][0].row_ptr_start = ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph));
        graph.dpu_param[0][0].col_idx_start = graph.dpu_param[0][0].row_ptr_start + static_cast<unsigned>(row_ptr_size * sizeof(uint32_t));
        graph.dpu_param[0][0].comp_start = graph.dpu_param[0][0].col_idx_start + static_cast<unsigned>(col_idx_size * sizeof(uint32_t));
        graph.dpu_param[0][0].flag_start = graph_dpu_param[0][0].comp_start + static_cast<unsigned>(comp_size * sizeof(uint32_t));

        csr.close();

    }
    else
        throw invalid_argument("Cannot open graph");

    return graph;
}

static Graph divide_graph(Graph& graph, uint32_t n) {
    Graph subgraph;

    for (uint32_t i = 0; i < n; i++) {
        subgraph.dpu_param.push_back(vector<DPUGraph> (1));
    }
    
    // distribute vertices in a balanced manner
    uint32_t num_v_origin = graph.dpu_param[0][0].num_v_origin;
    uint32_t q = num_v_origin / n;
    uint32_t r = num_v_origin - q * n;

    uint32_t *unit_v = new uint32_t[n];

    for (uint32_t i = 0; i < n; i++) {
        unit_v[i] = q;
    }
    for (uint32_t i = 0; i < r; i++) {
        unit_v[i]++;
    }

    uint32_t col_idx_max = 0;
    uint32_t row_start = 0;
    uint32_t row_end = 0;
    for (uint32_t i = 0; i < n; i++) {
        uint32_t num_e;

        row_end += unit_v[i];
        num_e = graph.row_ptr[0][row_end] - graph.row_ptr[0][row_start];

        if (col_idx_max < num_e)
            col_idx_max = num_e;

        subgraph.dpu_param[i][0].num_v_origin = num_v_origin;
        subgraph.dpu_param[i][0].num_e = num_e;
        subgraph.dpu_param[i][0].num_t = 1;
        subgraph.dpu_param[i][0].node_start_idx = row_start;

        row_start = row_end;
    }

    uint32_t row_ptr_size = ROUND_UP_TO_MULTIPLE_OF_2(unit_v[0]+1);
    uint32_t col_idx_size = ROUND_UP_TO_MULTIPLE_OF_2(col_idx_max);
    uint32_t comp_size = ROUND_UP_TO_MULTIPLE_OF_2(num_v_origin);

    for (uint32_t i = 0; i < n; i++) {
        subgraph.row_ptr.push_back(vector<uint32_t> (row_ptr_size));
        subgraph.col_idx.push_back(vector<uint32_t> (col_idx_size));
    }
    copy(graph.comp.begin(), graph.comp.end(), back_inserter(subgraph.comp));

    row_start = 0;
    row_end = 0;

    for (uint32_t i = 0; i < n; i++) {
        subgraph.dpu_param[i][0].num_v = unit_v[i];
        row_end += unit_v[i];

        uint32_t offset = graph.row_ptr[0][row_start];

        uint32_t idx = 0;
        for (uint32_t j = row_start; j <= row_end; j++) {
            subgraph.row_ptr[i][idx] = graph.row_ptr[0][j] - offset;
            idx++;
        }

        idx = 0;
        for (uint32_t j = graph.row_ptr[0][row_start]; j < graph.row_ptr[0][row_end]; j++) {
            subgraph.col_idx[i][idx] = graph.col_idx[0][j];
            idx++;
        }

        subgraph.dpu_param[i][0].row_ptr_start = ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph));
        subgraph.dpu_param[i][0].col_idx_start = subgraph.dpu_param[i][0].row_ptr_start + static_cast<unsigned>(row_ptr_size * sizeof(uint32_t));
        subgraph.dpu_param[i][0].comp_start = subgraph.dpu_param[i][0].col_idx_start + static_cast<unsigned>(col_idx_size * sizeof(uint32_t));
        subgraph.dpu_param[i][0].flag_start = subgraph_dpu_param[i][0].comp_start + static_cast<unsigned>(comp_size * sizeof(uint32_t));

        row_start = row_end;
    }

    delete [] unit_v;

    return subgraph;
}

static Graph divide_graph_improved(Graph& graph, uint32_t n) {
    Graph subgraph;

    for (uint32_t i = 0; i < n; i++) {
        subgraph.dpu_param.push_back(vector<DPUGraph> (1));
    }
    
    // distribute vertices in a balanced manner
    uint32_t num_v_origin = graph.dpu_param[0][0].num_v_origin;
    uint32_t num_e_origin = graph.dpu_param[0][0].num_e;
    uint32_t comp_size = ROUND_UP_TO_MULTIPLE_OF_2(num_v_origin);

    uint32_t *unit_v = new uint32_t[n];
    uint32_t unit_e = num_e_origin / n;

    uint32_t row_idx_max = 0;
    uint32_t col_idx_max = 0;
    uint32_t row_start = 0;
    uint32_t row_end = 0;

    for (uint32_t i = 0; i < n; i++) {
        uint32_t num_e = 0;
        uint32_t start = graph.row_ptr[0][row_start];
        while (num_e < unit_e && row_end < num_v_origin) {
            row_end++;

            /*
            // this dividing makes some dpus have no edges.
            // so, we think about the another options;
            // all of dpus has less edges than unit_e, and rest are processed in CPU.

            if (graph.row_ptr[0][row_end] - start > unit_e) {
                row_end--;
                break;
            }
            */

            num_e = graph.row_ptr[0][row_end] - start;
            if (row_end == num_v_origin)
                break;
        }

        unit_v[i] = row_end - row_start;
        if (row_idx_max < unit_v[i])
            row_idx_max = unit_v[i];

        if (col_idx_max < num_e)
            col_idx_max = num_e;
        
        subgraph.dpu_param[i][0].num_v_origin = num_v_origin;
        subgraph.dpu_param[i][0].num_e = num_e;
        subgraph.dpu_param[i][0].num_t = 1;
        subgraph.dpu_param[i][0].node_start_idx = row_start;

        row_start = row_end;
    }

    uint32_t row_ptr_size = ROUND_UP_TO_MULTIPLE_OF_2(row_idx_max);
    uint32_t col_idx_size = ROUND_UP_TO_MULTIPLE_OF_2(col_idx_max);
    uint32_t comp_size = ROUND_UP_TO_MULTIPLE_OF_2(num_v_origin);

    for (uint32_t i = 0; i < n; i++) {
        subgraph.row_ptr.push_back(vector<uint32_t> (row_ptr_size));
        subgraph.col_idx.push_back(vector<uint32_t> (col_idx_size));
    }
    copy(graph.comp.begin(), graph.comp.end(), back_inserter(subgraph.comp));

    row_start = 0;
    row_end = 0;

    for (uint32_t i = 0; i < n; i++) {
        subgraph.dpu_param[i][0].num_v = unit_v[i];
        row_end += unit_v[i];

        uint32_t offset = graph.row_ptr[0][row_start];

        uint32_t idx = 0;
        for (uint32_t j = row_start; j <= row_end; j++) {
            subgraph.row_ptr[i][idx] = graph.row_ptr[0][j] - offset;
            idx++;
        }

        idx = 0;
        for (uint32_t j = graph.row_ptr[0][row_start]; j < graph.row_ptr[0][row_end]; j++) {
            subgraph.col_idx[i][idx] = graph.col_idx[0][j];
            idx++;
        }

        subgraph.dpu_param[i][0].row_ptr_start = ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph));
        subgraph.dpu_param[i][0].col_idx_start = subgraph.dpu_param[i][0].row_ptr_start + static_cast<unsigned>(row_ptr_size * sizeof(uint32_t));
        subgraph.dpu_param[i][0].comp_start = subgraph.dpu_param[i][0].col_idx_start + static_cast<unsigned>(col_idx_size * sizeof(uint32_t));
        subgraph.dpu_param[i][0].flag_start = subgraph_dpu_param[i][0].comp_start + static_cast<unsigned>(comp_size * sizeof(uint32_t));

        row_start = row_end;
    }

    delete [] unit_v;

    return subgraph;
}

static void tiling(Graph& subgraph, uint32_t n, uint32_t t) {
    uint32_t num_v_origin = subgraph.dpu_param[0][0].num_v_origin;
    uint32_t q_t = num_v_origin / t;
    uint32_t q_r = num_v_origin - q_t * t;

    uint32_t *unit_t = new uint32_t[t];

    for (uint32_t i = 0; i < t; i++)
        unit_t[i] = q_t;
    for (uint32_t i = 0; i < q_r; i++)
        unit_t[i]++;
    for (uint32_t i = 1; i < t; i++)
        unit_t[i] += unit_t[i-1];

    uint32_t row_ptr_size = ROUND_UP_TO_MULTIPLE_OF_2(subgraph.dpu_param[0][0].num_v * t + 1);
    uint32_t cur_row_ptr_size = ROUND_UP_TO_MULTIPLE_OF_2(subgraph.dpu_param[0][0].num_v + 1);
    uint32_t offset = row_ptr_size - cur_row_ptr_size;

    vector<vector<uint32_t>> vertices;
    vector<vector<uint32_t>> edges;
    vector<uint32_t> edge_acm(t);

    for (uint32_t i = 0; i < t; i++) {
        vertices.push_back(vector<uint32_t> ());
        edges.push_back(vector<uint32_t> ());
    }

    for (uint32_t i = 0; i < n; i++) {
        subgraph.dpu_param[i][0].num_t = t;
        uint32_t row_end = subgraph.dpu_param[i][0].num_v;

        for (uint32_t j = 0; j < row_end; j++) {
            for (uint32_t k = subgraph.row_ptr[i][j]; k < subgraph.row_ptr[i][j+1]; k++) {
                uint32_t col = subgraph.col_idx[i][k];
                for (uint32_t l = 0; l < t; l++) {
                    if (col < unit_t[l]) {
                        edge_acm[l]++;
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

        subgraph.row_ptr[i].resize(row_ptr_size);
        uint32_t idx = 1;
        uint32_t v_acm = 0;
        for (uint32_t j = 0; j < t; j++) {
            for (uint32_t k = 0; k < vertices[j].size(); k++) {
                v_acm += vertices[j][k];
                subgraph.row_ptr[i][idx] = v_acm;
                idx++;
            }
        }

        idx = 0;
        for (uint32_t j = 0; j < t; j++) {
            for (uint32_t k = 0; k < edges[j].size(); k++) {
                subgraph.col_idx[i][idx] = edges[j][k];
                idx++;
            }
        }

        subgraph.dpu_param[i][0].col_idx_start += offset * sizeof(uint32_t);
        subgraph.dpu_param[i][0].comp_start += offset * sizeof(uint32_t);

        for (uint32_t j = 0; j < t; j++) {
            vertices[j].clear();
            edges[j].clear();
        }
    }

    delete [] unit_t;
}

#endif