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
    vector<DPUGraph> dpu_param(1, 0);
    dpu_param[0].num_v = graph.num_v;
    dpu_param[0].num_e = graph.num_e;
    dpu_param[0].row_ptr_start = ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph));
    dpu_param[0].col_idx_start = static_cast<unsigned>(graph.row_ptr.size() * 4);
    dpu_param[0].value_start = static_cast<unsigned>(graph.col_idx.size() * 4);
    dpu_param[0].output_start = static_cast<unsigned>(graph.value.size() * 4);

    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, 0, dpu_param, ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph)));
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, dpu_param[0].row_ptr_start, graph.row_ptr, static_cast<unsigned>(graph.row_ptr.size() * 4));
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, dpu_param[0].col_idx_start, graph.col_idx, static_cast<unsigned>(graph.col_idx.size() * 4));
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, dpu_param[0].value_start, graph.value, static_cast<unsigned>(graph.value.size() * 4));
}

void populate_mram(DpuSetOps& dpu, Graph& graph, uint32_t id) {
    // TO-DO
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

        cout<<"HOST ELAPSED TIME: "<<chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9 <<" secs."<<endl;

        // TO-DO : ours

    } catch (const DpuError & e) {
        cerr << e.what() << endl;
    }
}