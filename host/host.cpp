#include <dpu>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>
#include <sys/stat.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "../support/common.h"
#include "../support/graph.h"

using namespace dpu;
using namespace std;

#define NB_OF_DPUS 1

#ifndef DPU_BASELINE
#define DPU_BASELINE "pr_baseline"
#endif

#ifndef DPU_OURS
#define DPU_OURS "pr_ours"
#endif

DPUGraph populate_mram(DpuSetOps& dpu, Graph& graph) {
    dpu.copy("num_v", graph.num_v);
    dpu.copy("num_e", graph.num_e);
    dpu.copy("row_ptr", graph.row_ptr, static_cast<unsigned>(graph.num_v+1));
    dpu.copy("col_idx", graph.col_idx, static_cast<unsigned>(graph.num_e));
    dpu.copy("value", graph.value, static_cast<unsigned>(graph.num_e));

    return dpu_graph;
}

void populate_mram(DpuSetOps& dpu, Graph& graph, uint32_t id) {
    // TO-DO
}

int main(int argc, char** argv) {
    // read graph file
    string csr_path;
    string output_path;

    int opt = 0;
    while((opt = getopt(argc, argv, "i:o:"))) {
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

        auto system = DpuSet::allocate(NB_OF_DPUS);
        auto dpu_baseline = system.dpus()[0];
        dpu_baseline->load(DPU_BASELINE)
        populate_mram(*dpu_baseline);
        dpu->exec();

        // TO-DO : ours

    } catch (const DpuError & e) {
        cerr << e.what() << endl;
    }
}