#include <dpu>
#include <dpu_log.h>

#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <chrono>

#include <sys/stat.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "graph.h"
#include "../support/common.h"

using namespace std;

#ifndef DPU_BASELINE
#define DPU_BASELINE "./bin/pr_baseline"
#endif

#ifndef DPU_OURS
#define DPU_OURS "./bin/pr_ours"
#endif

void populate_mram(dpu_set_t& dpu, Graph& graph, uint32_t id) {
    uint32_t row_ptr_size = (graph.dpu_param[id].col_idx_start - graph.dpu_param[id].row_ptr_start) / sizeof(uint32_t);
    uint32_t col_idx_size = (graph.dpu_param[id].value_start - graph.dpu_param[id].col_idx_start) / sizeof(uint32_t);
    uint32_t value_size = (graph.dpu_param[id].out_deg_start - graph.dpu_param[id].value_start) / sizeof(float);
    uint32_t out_deg_size = (graph.dpu_param[id].output_start - graph.dpu_param[id].out_deg_start) / sizeof(uint32_t);

    DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME, 0, (uint8_t*)&graph.dpu_param[id], ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph))));
    DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[id].row_ptr_start, (uint8_t*)&graph.row_ptr[id*row_ptr_size], graph.dpu_param[id].col_idx_start - graph.dpu_param[id].row_ptr_start));
    DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[id].col_idx_start, (uint8_t*)&graph.col_idx[id*col_idx_size], graph.dpu_param[id].value_start - graph.dpu_param[id].col_idx_start));
    DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[id].value_start, (uint8_t*)graph.value, graph.dpu_param[id].out_deg_start - graph.dpu_param[id].value_start));
    DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[id].out_deg_start, (uint8_t*)graph.out_deg, graph.dpu_param[id].output_start - graph.dpu_param[id].out_deg_start));
}

void populate_mram(dpu_set_t& dpu, Graph_X graph, uint32_t id) {
    uint32_t row_ptr_size = (graph.dpu_param[id].col_idx_start - graph.dpu_param[id].row_ptr_start) / sizeof(uint32_t);
    uint32_t col_idx_size = (graph.dpu_param[id].feature_c_start - graph.dpu_param[id].col_idx_start) / sizeof(uint32_t);
    uint32_t feature_c_size = (graph.dpu_param[id].feature_r_start - graph.dpu_param[id].feature_c_start) / sizeof(Feature);
    uint32_t feature_r_size = (graph.dpu_param[id].output_start - graph.dpu_param[id].feature_r_start) / sizeof(Feature);

    DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME, 0, (uint8_t*)&graph.dpu_param[id], ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph_X))));
    DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[id].row_ptr_start, (uint8_t*)&graph.row_ptr[id*row_ptr_size], row_ptr_size * sizeof(uint32_t)));
    DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[id].col_idx_start, (uint8_t*)&graph.col_idx[id*col_idx_size], col_idx_size * sizeof(uint32_t)));
    DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[id].feature_c_start, (uint8_t*)graph.feature_c, feature_c_size * sizeof(Feature)));
    DPU_ASSERT(dpu_copy_to(dpu, DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[id].feature_r_start, (uint8_t*)&graph.feature_r[id*feature_r_size], feature_r_size * sizeof(Feature)));
}

void populate_mram_parallel(dpu_set_t& dpu_set, Graph& graph) {
    dpu_set_t dpu;

    uint32_t idx = 0;
    uint32_t row_ptr_size = (graph.dpu_param[0].col_idx_start - graph.dpu_param[0].row_ptr_start) / sizeof(uint32_t);
    uint32_t col_idx_size = (graph.dpu_param[0].value_start - graph.dpu_param[0].col_idx_start) / sizeof(uint32_t);
    uint32_t value_size = (graph.dpu_param[0].out_deg_start - graph.dpu_param[0].value_start) / sizeof(float);
    uint32_t out_deg_size = (graph.dpu_param[0].output_start - graph.dpu_param[0].out_deg_start) / sizeof(uint32_t);

    DPU_FOREACH(dpu_set, dpu, idx) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &graph.dpu_param[idx]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph)), DPU_XFER_DEFAULT));

    idx = 0;
    DPU_FOREACH(dpu_set, dpu, idx) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &graph.row_ptr[idx*row_ptr_size]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[0].row_ptr_start, row_ptr_size * sizeof(uint32_t), DPU_XFER_DEFAULT));

    idx = 0;
    DPU_FOREACH(dpu_set, dpu, idx) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &graph.col_idx[idx*col_idx_size]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[0].col_idx_start, col_idx_size * sizeof(uint32_t), DPU_XFER_DEFAULT));

    DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[0].value_start, (uint8_t*)graph.value, value_size * sizeof(float), DPU_XFER_DEFAULT));
    DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[0].out_deg_start, (uint8_t*)graph.out_deg, out_deg_size * sizeof(uint32_t), DPU_XFER_DEFAULT));
}

void populate_mram_parallel(dpu_set_t& dpu_set, Graph_X& graph) {
    dpu_set_t dpu;

    uint32_t idx = 0;
    uint32_t row_ptr_size = (graph.dpu_param[0].col_idx_start - graph.dpu_param[0].row_ptr_start) / sizeof(uint32_t);
    uint32_t col_idx_size = (graph.dpu_param[0].feature_c_start - graph.dpu_param[0].col_idx_start) / sizeof(uint32_t);
    uint32_t feature_c_size = (graph.dpu_param[0].feature_r_start - graph.dpu_param[0].feature_c_start) / sizeof(Feature);
    uint32_t feature_r_size = (graph.dpu_param[0].output_start - graph.dpu_param[0].feature_r_start) / sizeof(Feature);

    DPU_FOREACH(dpu_set, dpu, idx) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &graph.dpu_param[idx]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, 0, ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph_X)), DPU_XFER_DEFAULT));

    idx = 0;
    DPU_FOREACH(dpu_set, dpu, idx) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &graph.row_ptr[idx*row_ptr_size]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[0].row_ptr_start, row_ptr_size * sizeof(uint32_t), DPU_XFER_DEFAULT));

    idx = 0;
    DPU_FOREACH(dpu_set, dpu, idx) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &graph.col_idx[idx*col_idx_size]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[0].col_idx_start, col_idx_size * sizeof(uint32_t), DPU_XFER_DEFAULT));

    DPU_ASSERT(dpu_broadcast_to(dpu_set, DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[0].feature_c_start, (uint8_t*)graph.feature_c, feature_c_size * sizeof(Feature), DPU_XFER_DEFAULT));

    idx = 0;
    DPU_FOREACH(dpu_set, dpu, idx) {
        DPU_ASSERT(dpu_prepare_xfer(dpu, &graph.feature_r[idx*feature_r_size]));
    }
    DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_TO_DPU, DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[0].feature_r_start, feature_r_size * sizeof(Feature), DPU_XFER_DEFAULT));
}

int main(int argc, char** argv) {
    // read graph file
    string csr_path;
    string output_path;
    uint32_t num_t = 1;
    uint32_t hash_key = 64;

    chrono::steady_clock::time_point begin, end;

    int opt = 0;
    while((opt = getopt(argc, argv, "i:o:t:h:")) != EOF) {
        switch (opt) {
            case 'i':
                csr_path = optarg;
                break;
            case 'o':
                output_path = optarg;
                break;
            case 't':
                num_t = stoi(optarg);
                break;
            case 'h':
                hash_key = stoi(optarg);
                break;
            case '?':
                cout <<"WRONG OPTIONS"<<endl;
                break;
        }
    }

    Graph graph = read_csr(csr_path);
    cout<<"GRAPH READ COMPLETE"<<endl;

    dpu_set_t dpu_set, dpu;
    DPU_ASSERT(dpu_alloc(1, NULL, &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, DPU_BASELINE, NULL));

    cout<<"BASELINE PROGRAM ALLOCATED"<<endl;

    begin = chrono::steady_clock::now();
    populate_mram(dpu_set, graph, 0);
    end = chrono::steady_clock::now();
    cout<<"DATA TRANSFER TIME: "<<chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9 <<" secs"<<endl;
    begin = chrono::steady_clock::now();
    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
    end = chrono::steady_clock::now();
    cout<<"HOST ELAPSED TIME: "<<chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9 <<" secs."<<endl;
    DPU_FOREACH(dpu_set, dpu) {
        DPU_ASSERT(dpu_log_read(dpu, stdout));
        DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[0].output_start, (uint8_t*)graph.output, ROUND_UP_TO_MULTIPLE_OF_2(graph.dpu_param[0].num_v) * sizeof(float)));
    }
    cout<<"OUTPUT RECEIVED"<<endl;
    for (uint32_t i = 0; i < 10; i++)
        cout<<"DPU RESULT: "<<graph.output[i]<<endl;

    DPU_ASSERT(dpu_free(dpu_set));

    // Ours
    Graph subgraph = divide_graph_naive(graph, NR_DPUS, num_t);

    DPU_ASSERT(dpu_alloc(NR_DPUS, NULL, &dpu_set));
    DPU_ASSERT(dpu_load(dpu_set, DPU_OURS, NULL));

    cout<<"OURS PROGRAM ALLOCATED"<<endl;

    begin = chrono::steady_clock::now();
    uint32_t idx = 0;
    DPU_FOREACH(dpu_set, dpu, idx) {
        populate_mram(dpu, subgraph, idx);
    }
    end = chrono::steady_clock::now();
    cout<<"DATA TRANSFER TIME: "<<chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9 <<" secs"<<endl;
    begin = chrono::steady_clock::now();
    DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
    end = chrono::steady_clock::now();
    cout<<"HOST ELAPSED TIME: "<<chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9 <<" secs."<<endl;
    idx = 0;
    uint32_t output_size = ROUND_UP_TO_MULTIPLE_OF_2(subgraph.dpu_param[0].num_v);
    DPU_FOREACH(dpu_set, dpu, idx) {
        DPU_ASSERT(dpu_log_read(dpu, stdout));
        DPU_ASSERT(dpu_copy_from(dpu, DPU_MRAM_HEAP_POINTER_NAME, subgraph.dpu_param[idx].output_start, (uint8_t*)&subgraph.output[idx * output_size], output_size * sizeof(float)));
    }
    cout<<"OUTPUT RECEIVED"<<endl;
    for (uint32_t i = 0; i < NR_DPUS; i++) {
        cout<<"DPU "<<i<<endl;
        for (uint32_t j = 0; j < 10; j++) {
            cout<<"DPU RESULT: "<<subgraph.output[i*output_size+j]<<endl;
        }
        cout<<endl;
    }

    // OURS (parallel)

    // cout<<"OURS PARALLEL START"<<endl;
    // begin = chrono::steady_clock::now();
    // populate_mram_parallel(dpu_set, subgraph);
    // end = chrono::steady_clock::now();
    // cout<<"DATA TRANSFER TIME: "<<chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9 <<" secs"<<endl;

    // begin = chrono::steady_clock::now();
    // DPU_ASSERT(dpu_launch(dpu_set, DPU_SYNCHRONOUS));
    // end = chrono::steady_clock::now();
    // cout<<"HOST ELAPSED TIME: "<<chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9 <<" secs."<<endl;

    // idx = 0;
    // DPU_FOREACH(dpu_set, dpu, idx) {
    //     DPU_ASSERT(dpu_prepare_xfer(dpu, &subgraph.output[idx*output_size]));
    // }
    // DPU_ASSERT(dpu_push_xfer(dpu_set, DPU_XFER_FROM_DPU, DPU_MRAM_HEAP_POINTER_NAME, subgraph.dpu_param[0].output_start, output_size * sizeof(float), DPU_XFER_DEFAULT));

    // cout<<"OUTPUT RECEIVED"<<endl;
    // for (uint32_t i = 0; i < NR_DPUS; i++) {
    //     cout<<"DPU "<<i<<endl;
    //     for (uint32_t j = 0; j < 10; j++) {
    //         cout<<"DPU RESULT: "<<subgraph.output[i*output_size+j]<<endl;
    //     }
    //     cout<<endl;
    // }

    // Graph_X subgraph_r = divide_graph_ours(subgraph, NR_DPUS);


    // cout<<"TEST_1"<<endl;
    // begin = chrono::steady_clock::now();
    // uint32_t idx = 0;
    // DPU_FOREACH(dpu_set, dpu, idx) {
    //     populate_mram(dpu, subgraph_r, idx);
    // }
    // end = chrono::steady_clock::now();
    // cout<<"DATA TRANSFER TIME: "<<chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9 <<" secs"<<endl;

    // cout<<"TEST_2"<<endl;
    // begin = chrono::steady_clock::now();
    // populate_mram_parallel(dpu_set, subgraph_r);
    // end = chrono::steady_clock::now();
    // cout<<"DATA TRANSFER TIME: "<<chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9 <<" secs"<<endl;

    free_graph(graph);
    free_graph(subgraph);
    // free_graph(subgraph_r);
}
