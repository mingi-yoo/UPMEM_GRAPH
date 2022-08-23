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

#include "graph.h"

using namespace dpu;
using namespace std;

#ifndef DPU_BASELINE
#define DPU_BASELINE "./bin/pr_baseline"
#endif

#ifndef DPU_OURS
#define DPU_OURS "./bin/pr_ours"
#endif

void populate_mram(DpuSetOps& dpu, Graph& graph) {
    vector<uint32_t> g_info(2, 0);
    g_info[0] = graph.num_v;
    g_info[1] = graph.num_e;
    dpu.copy("g_info", g_info, static_cast<unsigned>(2 * 4));
    dpu.copy("row_ptr", graph.row_ptr, static_cast<unsigned>(graph.row_ptr.size() * 4));
    dpu.copy("col_idx", graph.col_idx, static_cast<unsigned>(graph.col_idx.size() * 4));
    dpu.copy("value", graph.value, static_cast<unsigned>(graph.value.size() * 4));
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

        auto system = DpuSet::allocate(NR_DPUS);
        auto dpu_baseline = system.dpus()[0];
        dpu_baseline->load(DPU_BASELINE);
        populate_mram(*dpu_baseline, graph);
        dpu_baseline->exec();
        dpu_baseline->log(cout);

        // TO-DO : ours

    } catch (const DpuError & e) {
        cerr << e.what() << endl;
    }
}