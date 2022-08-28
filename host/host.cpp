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

using namespace dpu;
using namespace std;

#ifndef DPU_BASELINE
#define DPU_BASELINE "./bin/pr_baseline"
#endif

#ifndef DPU_OURS
#define DPU_OURS "./bin/pr_ours"
#endif

void populate_mram(DpuSetOps& dpu, Graph& graph) {
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, 0, graph.dpu_param[0], ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph)));
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[0][0].row_ptr_start, graph.row_ptr[0]);
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[0][0].col_idx_start, graph.col_idx[0]);
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[0][0].value_start, graph.value[0]);
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[0][0].out_deg_start, graph.out_deg[0]);
}

void populate_mram(DpuSetOps& dpu, Graph& graph, uint32_t id) {
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, 0, graph.dpu_param[id], ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph)));
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[id][0].row_ptr_start, graph.row_ptr[id]);
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[id][0].col_idx_start, graph.col_idx[id]);
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[id][0].value_start, graph.value[id]);
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[id][0].out_deg_start, graph.out_deg[id]);
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

    chrono::steady_clock::time_point begin, end;

    int opt = 0;
    while((opt = getopt(argc, argv, "i:o:")) != EOF) {
        switch (opt) {
            case 'i':
                csr_path = optarg;
                break;
            case 'o':
                output_path = optarg;
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

        dpu_baseline->load(DPU_BASELINE);
        begin = chrono::steady_clock::now();
        populate_mram(*dpu_baseline, graph);
        end = chrono::steady_clock::now();
        cout<<"DATA TRANSFER TIME: "<<chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9 <<" secs"<<endl;
        begin = chrono::steady_clock::now();
        dpu_baseline->exec();
        end = chrono::steady_clock::now();
        dpu_baseline->log(cout);

        vector<vector<float>> result(NR_DPUS);
        for (uint32_t i = 0; i < NR_DPUS; i++)
            result[i].resize(static_cast<unsigned>(graph.value[0].size()));

        dpu_baseline->copy(result, static_cast<unsigned>(graph.value[0].size() * 4), DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[0][0].output_start);

        cout<<"HOST ELAPSED TIME: "<<chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9 <<" secs."<<endl;

        cout<<"PR CHECK"<<endl;

        for (uint32_t i = 0; i < 5; i++)
            cout<< result[0][i] <<endl;

        cout<<endl;

        // TO-DO : ours
        Graph subgraphs = divide_graph(graph, NR_DPUS);

        system.load(DPU_OURS);
        cout<<"OURS PROGRAM ALLOCATED"<<endl;

        begin = chrono::steady_clock::now();
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            auto dpu = system.dpus()[i];
            populate_mram(*dpu, subgraphs, i);
        }
        end = chrono::steady_clock::now();
        cout<<"DATA TRANSFER TIME: "<<chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9 <<" secs"<<endl;

        begin = chrono::steady_clock::now();
        system.exec();
        end = chrono::steady_clock::now();
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            auto dpu = system.dpus()[i];
            dpu->log(cout);
        }

        cout<<"PR CHECK"<<endl;
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            auto dpu = system.dpus()[i];
            dpu->copy(result, static_cast<unsigned>(subgraphs.value[i].size() * 4), DPU_MRAM_HEAP_POINTER_NAME, subgraphs.dpu_param[i][0].output_start);
            for (uint32_t j = 0; j < 5; j++) {
                cout<<result[0][j]<<endl;
            }
            cout<<endl;
        }

        cout<<"HOST ELAPSED TIME: "<<chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9 <<" secs."<<endl;

        cout<<"OURS PROGRAM ALLOCATED (PARALLEL)"<<endl;
        begin = chrono::steady_clock::now();
        populate_mram_parallel(system, subgraphs);
        end = chrono::steady_clock::now();

        cout<<"DATA TRANSFER TIME: "<<chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9 <<" secs"<<endl;

        begin = chrono::steady_clock::now();
        system.exec();
        end = chrono::steady_clock::now();
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            auto dpu = system.dpus()[i];
            dpu->log(cout);
        }

        cout<<"PR CHECK"<<endl;
        system.copy(result, static_cast<unsigned>(subgraphs.value[0].size() * 4), DPU_MRAM_HEAP_POINTER_NAME, subgraphs.dpu_param[0][0].output_start);
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            for (uint32_t j = 0; j < 5; j++) {
                cout<<result[i][j]<<endl;
            }
            cout<<endl;
        }

        cout<<"HOST ELAPSED TIME: "<<chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9 <<" secs."<<endl;


    } catch (const DpuError & e) {
        cerr << e.what() << endl;
    }
}
