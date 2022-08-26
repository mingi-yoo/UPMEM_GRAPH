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

vector<DPUGraph> dpu_param(NR_DPUS);

void populate_mram(DpuSetOps& dpu, Graph& graph) {
    dpu_param[0].num_v_origin = graph.num_v_origin;
    dpu_param[0].num_v = graph.num_v;
    dpu_param[0].num_e = graph.num_e;
    dpu_param[0].row_ptr_start = ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph));
    dpu_param[0].col_idx_start = dpu_param[0].row_ptr_start + static_cast<unsigned>(graph.row_ptr.size() * 4);
    dpu_param[0].value_start = dpu_param[0].col_idx_start + static_cast<unsigned>(graph.col_idx.size() * 4);
    dpu_param[0].out_deg_start = dpu_param[0].value_start + static_cast<unsigned>(graph.out_deg.size() * 4);
    dpu_param[0].output_start = dpu_param[0].out_deg_start + static_cast<unsigned>(graph.value.size() * 4);

    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, 0, dpu_param, ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph)));
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, dpu_param[0].row_ptr_start, graph.row_ptr, static_cast<unsigned>(graph.row_ptr.size() * 4));
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, dpu_param[0].col_idx_start, graph.col_idx, static_cast<unsigned>(graph.col_idx.size() * 4));
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, dpu_param[0].value_start, graph.value, static_cast<unsigned>(graph.value.size() * 4));
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, dpu_param[0].out_deg_start, graph.out_deg, static_cast<unsigned>(graph.out_deg.size() * 4));
}

void populate_mram(DpuSetOps& dpu, Graph& graph, uint32_t id) {
    dpu_param[id].num_v_origin = graph.num_v_origin;
    dpu_param[id].num_v = graph.num_v;
    dpu_param[id].num_e = graph.num_e;
    dpu_param[id].row_ptr_start = ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph));
    dpu_param[id].col_idx_start = dpu_param[id].row_ptr_start + static_cast<unsigned>(graph.row_ptr.size() * 4);
    dpu_param[id].value_start = dpu_param[id].col_idx_start + static_cast<unsigned>(graph.col_idx.size() * 4);
    dpu_param[id].out_deg_start = dpu_param[id].value_start + static_cast<unsigned>(graph.out_deg.size() * 4);
    dpu_param[id].output_start = dpu_param[id].out_deg_start + static_cast<unsigned>(graph.value.size() * 4);

    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, 0, dpu_param, ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph)));
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, dpu_param[id].row_ptr_start, graph.row_ptr, static_cast<unsigned>(graph.row_ptr.size() * 4));
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, dpu_param[id].col_idx_start, graph.col_idx, static_cast<unsigned>(graph.col_idx.size() * 4));
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, dpu_param[id].value_start, graph.value, static_cast<unsigned>(graph.value.size() * 4));
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, dpu_param[id].out_deg_start, graph.out_deg, static_cast<unsigned>(graph.out_deg.size() * 4));
}

int main(int argc, char** argv) {
    // read graph file
    string csr_path;
    string output_path;

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
        populate_mram(*dpu_baseline, graph);
        chrono::steady_clock::time_point begin = chrono::steady_clock::now();
        dpu_baseline->exec();
        chrono::steady_clock::time_point end = chrono::steady_clock::now();
        dpu_baseline->log(cout);

        vector<vector<float>> result(1);
        result.front().resize(static_cast<unsigned>(graph.value.size()));
        dpu_baseline->copy(result, static_cast<unsigned>(graph.value.size() * 4), DPU_MRAM_HEAP_POINTER_NAME, dpu_param[0].output_start);

        cout<<"HOST ELAPSED TIME: "<<chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9 <<" secs."<<endl;

        cout<<"PR CHECK"<<endl;

        for (int i = 0; i < 5; i++)
            cout<< result.front()[i] <<endl;

        // TO-DO : ours
        vector<Graph> subgraphs = divide_graph(graph, NR_DPUS);

        system.load(DPU_OURS);
        cout<<"OURS PROGRAM ALLOCATED"<<endl;

        for (uint32_t i = 0; i < NR_DPUS; i++) {
            auto dpu = system.dpus()[i];
            populate_mram(*dpu, subgraphs[i], i);
        }
        begin = chrono::steady_clock::now();
        system.exec();
        end = chrono::steady_clock::now();
        for (uint32_t i = 0; i < NR_DPUS; i++) {
            auto dpu = system.dpus()[i];
            dpu->log(cout);
        }

        cout<<"HOST ELAPSED TIME: "<<chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9 <<" secs."<<endl;


    } catch (const DpuError & e) {
        cerr << e.what() << endl;
    }
}
