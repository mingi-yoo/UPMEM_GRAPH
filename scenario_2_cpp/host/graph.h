#ifndef _GRAPH_H_
#define _GRAPH_H_

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <iterator>
#include <algorithm>

#include "../support/common.h"

using namespace std;

struct Graph {
    vector<vector<DPUGraph>> dpu_param;
    vector<uint32_t> hash_fc;
    vector<vector<uint32_t>> hash_fr;
    vector<vector<uint32_t>> row_ptr;
    vector<vector<uint32_t>> col_idx;
    vector<Feature> fc;
    vector<vector<Feature>> fr;
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
        uint32_t feature_size = ROUND_UP_TO_MULTIPLE_OF_2(graph.dpu_param[0][0].num_v);

        graph.row_ptr.push_back(vector<uint32_t> (row_ptr_size));
        graph.col_idx.push_back(vector<uint32_t> (col_idx_size));
        graph.fc.resize(feature_size);

        for (uint32_t i = 0; i <= graph.dpu_param[0][0].num_v; i++) {
            csr >> graph.row_ptr[0][i];
        }

        for (uint32_t i = 0; i < graph.dpu_param[0][0].num_e; i++) {
            csr >> graph.col_idx[0][i];
        }

        for (uint32_t i = 0; i < graph.dpu_param[0][0].num_v; i++) {
            graph.fc[i].v_id = i;
            csr >> graph.fc[i].out_deg;
            graph.fc[i].value = 1.0f / graph.dpu_param[0][0].num_v;
        };

        // set offset
        graph.dpu_param[0][0].row_ptr_start = ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph));
        graph.dpu_param[0][0].col_idx_start = graph.dpu_param[0][0].row_ptr_start + static_cast<unsigned>(row_ptr_size * sizeof(uint32_t));
        graph.dpu_param[0][0].fc_start = graph.dpu_param[0][0].col_idx_start + static_cast<unsigned>(col_idx_size * sizeof(uint32_t));
        graph.dpu_param[0][0].output_start = graph.dpu_param[0][0].fc_start + static_cast<unsigned>(feature_size * sizeof(Feature));

        csr.close();

    }
    else
        throw invalid_argument("Cannot open graph");

    return graph;
}

static Graph divide_graph(Graph& graph, uint32_t n, uint32_t t) {
    Graph subgraph;

    for (uint32_t i = 0; i < n; i++) {
        subgraph.dpu_param.push_back(vector<DPUGraph> (1));
    }
    
    // distribute vertices in a balanced manner
    uint32_t num_v_origin = graph.dpu_param[0][0].num_v_origin;
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
    for (uint32_t i = 1; i < t; i++) {
        unit_t[i] += unit_t[i-1];
    }

    cout<<"partitioned vertex check"<<endl;
    for (uint32_t i = 0; i < n; i++) {
        cout<<unit_v[i]<<" ";
    }
    cout<<endl;
    cout<<"partitioned tile check"<<endl;
    for (uint32_t i = 0; i < t; i++) {
        cout<<unit_t[i]<<" ";
    }
    cout<<endl;

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
        subgraph.dpu_param[i][0].num_t = t;

        row_start = row_end;
    }

    uint32_t row_ptr_size = ROUND_UP_TO_MULTIPLE_OF_2(unit_v[0]*t+1);
    uint32_t col_idx_size = ROUND_UP_TO_MULTIPLE_OF_2(col_idx_max);
    uint32_t feature_size = ROUND_UP_TO_MULTIPLE_OF_2(num_v_origin);

    for (uint32_t i = 0; i < n; i++) {
        subgraph.row_ptr.push_back(vector<uint32_t> (row_ptr_size));
        subgraph.col_idx.push_back(vector<uint32_t> (col_idx_size));
    }
    copy(graph.fc.begin(), graph.fc.end(), back_inserter(subgraph.fc));

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
        subgraph.dpu_param[i][0].num_v = unit_v[i];
        row_end += unit_v[i];

        subgraph.row_ptr[i][0] = 0;

        for (uint32_t j = row_start; j < row_end; j++) {
            for (uint32_t k = graph.row_ptr[0][j]; k < graph.row_ptr[0][j+1]; k++) {
                uint32_t col = graph.col_idx[0][k]; 
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

        subgraph.dpu_param[i][0].row_ptr_start = ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph));
        subgraph.dpu_param[i][0].col_idx_start = subgraph.dpu_param[i][0].row_ptr_start + static_cast<unsigned>(row_ptr_size * sizeof(uint32_t));
        subgraph.dpu_param[i][0].fc_start = subgraph.dpu_param[i][0].col_idx_start + static_cast<unsigned>(col_idx_size * sizeof(uint32_t));
        subgraph.dpu_param[i][0].output_start = subgraph.dpu_param[i][0].fc_start + static_cast<unsigned>(feature_size * sizeof(Feature));

        row_start = row_end;

        for (uint32_t j = 0; j < t; j++) {
            vertices[j].clear();
            edges[j].clear();
        }
    }

    delete [] unit_t;
    delete [] unit_v;

    return subgraph;
}

static void divide_feature(Graph& subgraph, uint32_t n, uint32_t hash_key) {
    vector<Feature> fc_new;
    vector<vector<bool>> e_check;

    uint32_t num_v_origin = subgraph.dpu_param[0][0].num_v_origin;

    for (uint32_t i = 0; i < n; i++) {
        subgraph.dpu_param[i][0].hash_key = hash_key;
        e_check.push_back(vector<bool> (num_v_origin));
    }

    for (uint32_t i = 0; i < n; i++) {
        for (uint32_t j = 0; j < subgraph.dpu_param[i][0].num_e; j++) {
            e_check[i][subgraph.col_idx[i][j]] = true;
        }
    }

    vector<vector<uint32_t>> col_c;
    vector<vector<vector<uint32_t>>> col_r;

    for (uint32_t i = 0; i < hash_key; i++)
        col_c.push_back(vector<uint32_t> ());

    for (uint32_t i = 0; i < n; i++) {
        col_r.push_back(vector<vector<uint32_t>> ());
        for (uint32_t j = 0; j < hash_key; j++)
            col_r[i].push_back(vector<uint32_t> ());
    }

    // check common col
    for (uint32_t i = 0; i < num_v_origin; i++) {
        bool check = true;
        for (uint32_t j = 0; j < n; j++) {
            if (!e_check[j][i]) {
                check = false;
                break;
            }
        }
        uint32_t hash_idx = i % hash_key;

        if (check) {
            col_c[hash_idx].push_back(i);
        }
        else {
            for (uint32_t j = 0; j < n; j++) {
                if (e_check[j][i])
                    col_r[j][hash_idx].push_back(i);
            }
        }
    }

    uint32_t fc_size = 0;
    uint32_t fr_size = 0;

    for (uint32_t i = 0; i < hash_key; i++)
        fc_size += col_c[i].size();
    fc_size = ROUND_UP_TO_MULTIPLE_OF_2(fc_size);

    for (uint32_t i = 0; i < n; i++) {
        uint32_t comp = 0;
        for (uint32_t j = 0; j < hash_key; j++)
            comp += col_r[i][j].size();
        if (comp > fr_size)
            fr_size = comp;
    }
    fr_size = ROUND_UP_TO_MULTIPLE_OF_2(fr_size);

    uint32_t hash_size = ROUND_UP_TO_MULTIPLE_OF_2(hash_key+1);
    subgraph.hash_fc.resize(hash_size);
    for (uint32_t i = 0; i < n; i++)
        subgraph.hash_fr.push_back(vector<uint32_t> (hash_size));

    fc_new.resize(fc_size);
    for (uint32_t i = 0; i < n; i++)
        subgraph.fr.push_back(vector<Feature> (fr_size));

    uint32_t edge_acm = 0;
    for (uint32_t i = 0; i < hash_key; i++) {
        subgraph.hash_fc[i] = edge_acm;
        for (uint32_t j = 0; j < col_c[i].size(); j++) {
            fc_new[edge_acm].v_id = col_c[i][j];
            fc_new[edge_acm].out_deg = subgraph.fc[col_c[i][j]].out_deg;
            fc_new[edge_acm].value = subgraph.fc[col_c[i][j]].value;
            edge_acm++;
        }
    }
    subgraph.hash_fc[hash_key] = edge_acm;

    for (uint32_t i = 0; i < n; i++) {
        edge_acm = 0;
        for (uint32_t j = 0; j < hash_key; j++) {
            subgraph.hash_fr[i][j] = edge_acm;
            for (uint32_t k = 0; k < col_r[i][j].size(); k++) {
                subgraph.fr[i][edge_acm].v_id = col_r[i][j][k];
                subgraph.fr[i][edge_acm].out_deg = subgraph.fc[col_r[i][j][k]].out_deg;
                subgraph.fr[i][edge_acm].value = subgraph.fc[col_r[i][j][k]].value;
                edge_acm++;
            }
        }
        subgraph.hash_fr[i][hash_key] = edge_acm;
    }

    subgraph.fc.swap(fc_new);

    uint32_t row_ptr_size = subgraph.row_ptr[0].size();
    uint32_t col_idx_size = subgraph.col_idx[0].size();

    for (uint32_t i = 0; i < n; i++) {
        subgraph.dpu_param[i][0].hash_fc_start = ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph));
        subgraph.dpu_param[i][0].hash_fr_start = subgraph.dpu_param[i][0].hash_fc_start + static_cast<unsigned>(hash_size * sizeof(uint32_t));
        subgraph.dpu_param[i][0].row_ptr_start = subgraph.dpu_param[i][0].hash_fr_start + static_cast<unsigned>(hash_size * sizeof(uint32_t));
        subgraph.dpu_param[i][0].col_idx_start = subgraph.dpu_param[i][0].row_ptr_start + static_cast<unsigned>(row_ptr_size * sizeof(uint32_t));
        subgraph.dpu_param[i][0].fc_start = subgraph.dpu_param[i][0].col_idx_start + static_cast<unsigned>(col_idx_size * sizeof(uint32_t));
        subgraph.dpu_param[i][0].fr_start = subgraph.dpu_param[i][0].fc_start + static_cast<unsigned>(fc_size * sizeof(Feature));
        subgraph.dpu_param[i][0].output_start = subgraph.dpu_param[i][0].fr_start + static_cast<unsigned>(fr_size * sizeof(Feature));
    }

    for (uint32_t i = 0; i < n; i++) {
        cout<<"DPU "<<i<<endl;
        for (uint32_t j = 0; j < 64; j++) {
            cout<<"hash info: "<<subgraph.hash_fc[j]<<", "<<subgraph.hash_fr[i][j]<<", "<<subgraph.fc[j].v_id<<", "<<subgraph.fr[i][j].v_id<<endl;
        }
    }
}

static void check_integrity(Graph& subgraph, uint32_t n, uint32_t hash_key) {
    // First check, all of vector size is equal
    cout<<"First check, all of vector size is equal..."<<endl;
    for (uint32_t i = 0; i < n; i++) {
        cout<<"Check "<<i<<endl;
        cout<<"------------------------"<<endl;
        cout<<"hash_fc: "<<subgraph.hash_fc.size()<<" "<< (subgraph.dpu_param[i][0].hash_fr_start - subgraph.dpu_param[i][0].hash_fc_start) / sizeof(uint32_t) << endl;
        cout<<"hash_fr: "<<subgraph.hash_fr[i].size()<<" "<< (subgraph.dpu_param[i][0].row_ptr_start - subgraph.dpu_param[i][0].hash_fr_start) / sizeof(uint32_t) << endl;
        cout<<"fc: "<<subgraph.fc.size()<<" "<< (subgraph.dpu_param[i][0].fr_start - subgraph.dpu_param[i][0].fc_start) / sizeof(Feature) << endl;
        cout<<"fr: "<<subgraph.fr[i].size()<<" "<< (subgraph.dpu_param[i][0].output_start - subgraph.dpu_param[i][0].fr_start) / sizeof(Feature) << endl;
        cout<<"------------------------"<<endl;
    }

    cout<<"Second check, all of feature can include all edges..."<<endl;
    for (uint32_t i = 0; i < n; i++) {
        cout<<"Check "<<i<<endl;
        vector<bool> check(subgraph.dpu_param[i][0].num_v_origin);
        uint32_t f_cnt = 0;
        uint32_t total_f = subgraph.hash_fc[hash_key] + subgraph.hash_fr[i][hash_key];

        for (uint32_t j = 0; j < subgraph.dpu_param[i][0].num_e; j++)
            check[subgraph.col_idx[i][j]] = true;
        for (uint32_t j = 0; j < check.size(); j++) {
            if (check[j])
                f_cnt++;
        }

        if (f_cnt != total_f) {
            cout<<"Number of Features is different: "<<f_cnt<<", "<<total_f<<endl;
            continue;
        }

        for (uint32_t j = 0; j < subgraph.hash_fc[hash_key]; j++)
            check[subgraph.fc[j].v_id] = false;

        for (uint32_t j = 0; j < subgraph.hash_fr[i][hash_key]; j++)
            check[subgraph.fr[i][j].v_id] = false;

        for (uint32_t j = 0; j < check.size(); j++) {
            if (check[j]) {
                cout<<"Feature cannot include all of edges!"<<endl;
                break;
            }
        }
    }

    cout<<"Check End"<<endl;
}

#endif