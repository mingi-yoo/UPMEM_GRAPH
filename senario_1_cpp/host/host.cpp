#include <dpu>
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

void populate_mram(DpuSetOps& dpu, Graph& graph, uint32_t id) {
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, 0, graph.dpu_param[id], ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph)));
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[id][0].row_ptr_start, graph.row_ptr[id]);
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[id][0].col_idx_start, graph.col_idx[id]);
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[id][0].value_start, graph.value);
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[id][0].out_deg_start, graph.out_deg);
}

void populate_mram_parallel(DpuSetOps& dpu, Graph& graph) {
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, 0, graph.dpu_param, ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph)));
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[0][0].row_ptr_start, graph.row_ptr);
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[0][0].col_idx_start, graph.col_idx);
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[0][0].value_start, graph.value);
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[0][0].out_deg_start, graph.out_deg);
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
    try {
        Graph graph = read_csr(csr_path);
        cout<<"GRAPH READ COMPLETE"<<endl;

        auto system = DpuSet::allocate(NR_DPUS);
        auto dpu_baseline = system.dpus()[0];

        cout<<"BASELINE PROGRAM ALLOCATED"<<endl;

        begin = chrono::steady_clock::now();
        populate_mram(*dpu_baseline, graph, 0);
        end = chrono::steady_clock::now();
        cout<<"DATA TRANSFER TIME: "<<chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9 <<" secs"<<endl;
        begin = chrono::steady_clock::now();
        dpu_baseline->exec();
        end = chrono::steady_clock::now();
        cout<<"HOST ELAPSED TIME: "<<chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9 <<" secs."<<endl;
        dpu_baseline->log(cout);

        vector<vector<float>> result(NR_DPUS);
        for (uint32_t i = 0; i < NR_DPUS; i++)
            result[i].resize(static_cast<unsigned>(graph.value.size()));

        dpu_baseline->copy(result, static_cast<unsigned>(graph.value.size() * sizeof(float)), DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[0][0].output_start);
        cout<<"OUTPUT RECEIVED"<<endl;
        for (uint32_t i = 0; i < 10; i++)
            cout<<"DPU RESULT: "<<result[0][i]<<endl;

        // Ours
        Graph subgraph = divide_graph(graph, NR_DPUS, num_t);

        system.load(DPU_OURS)

        cout<<"OURS PROGRAM ALLOCATED"<<endl;

        begin = chrono::steady_clock::now();
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            auto dpu = system.dpus()[i];
            populate_mram(*dpu, subgraph, i);
        }
        end = chrono::steady_clock::now();
        cout<<"DATA TRANSFER TIME: "<<chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9 <<" secs"<<endl;
        begin = chrono::steady_clock::now();
        system.exec();
        end = chrono::steady_clock::now();
        cout<<"HOST ELAPSED TIME: "<<chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9 <<" secs."<<endl;
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            auto dpu = system.dpus()[i];
            dpu->log(cout);
        }
        cout<<"OUTPUT RECEIVED"<<endl;
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            auto dpu = system.dpus()[i];
            dpu->copy(result, static_cast<unsigned>(subgraph.value.size() * sizeof(float)), DPU_MRAM_HEAP_POINTER_NAME, subgraph.dpu_param[i][0].output_start);
            cout<<"DPU "<<i<<endl;
            for (uint32_t j = 0; j < 5; j++) {
                cout<<result[0][j]<<endl;
            }
            cout<<endl;
        }

        // OURS (parallel)

        cout<<"OURS PARALLEL START"<<endl;
        begin = chrono::steady_clock::now();
        populate_mram_parallel(system, subgraph);
        end = chrono::steady_clock::now();
        cout<<"DATA TRANSFER TIME: "<<chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9 <<" secs"<<endl;

        begin = chrono::steady_clock::now();
        system.exec();
        end = chrono::steady_clock::now();
        cout<<"HOST ELAPSED TIME: "<<chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9 <<" secs."<<endl;
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            auto dpu = system.dpus()[i];
            dpu->log(cout);
        }
        system.copy(result, static_cast<unsigned>(subgraph.value.size() * sizeof(float)), DPU_MRAM_HEAP_POINTER_NAME, subgraph.dpu_param[0][0].output_start);

        cout<<"OUTPUT RECEIVED"<<endl;
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            cout<<"DPU "<<i<<endl;
            for (uint32_t j = 0; j < 5; j++) {
                cout<<result[i][j]<<endl;
            }
            cout<<endl;
        }
    } catch (const DpuError & e) {
        cerr << e.what() << endl;
    }

    return 0;
}
