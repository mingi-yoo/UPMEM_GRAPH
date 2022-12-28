#include <dpu>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <map>
#include <limits>
#include <chrono>

#include <sys/stat.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "graph.h"
#include "../support/common.h"

using namespace std;
using namespace dpu;

#ifndef DPU_OURS
#define DPU_OURS "./bin/cc"
#endif

struct TimeRecord {
    double transfer;
    double run;
    double output_return;
    double total;
};

void print_time(TimeRecord& time_base, TimeRecord& time_ours);

void populate_mram_parallel(DpuSetOps& dpu, Graph& graph) {
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, 0, graph.dpu_param, ROUND_UP_TO_MULTIPLE_OF_8(sizeof(DPUGraph)));
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[0][0].row_ptr_start, graph.row_ptr);
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[0][0].col_idx_start, graph.col_idx);
}

void copy_comp_to_dpu(DpuSetOps& dpu, Graph& graph) {
    dpu.copy(DPU_MRAM_HEAP_POINTER_NAME, graph.dpu_param[0][0].comp_start, graph.comp[0]);
}

void run_async(DpuSet& system, unsigned dummy) {
    system.exec();
}

int main(int argc, char** argv) {
    // read graph file
    string csr_path;
    string output_path;
    uint32_t num_t = 1;

    TimeRecord time_ours;

    chrono::steady_clock::time_point begin, end;

    int opt = 0;
    while((opt = getopt(argc, argv, "i:o:t:")) != EOF) {
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
            case '?':
                cout <<"WRONG OPTIONS"<<endl;
                break;
        }
    }
    try {
        Graph graph = read_csr(csr_path);
        cout<<"GRAPH READ COMPLETE"<<endl;

        auto system = DpuSet::allocate(NR_DPUS);
        
        vector<vector<uint32_t>> result(NR_DPUS);
        for (uint32_t i = 0; i < NR_DPUS; i++)
            result[i].resize(2);

        // Graph subgraph = divide_graph(graph, NR_DPUS);
        Graph subgraph = divide_graph_improved(graph, NR_DPUS);

        if (num_t > 1)
            tiling(subgraph, NR_DPUS, num_t);

        system.load(DPU_OURS);

        cout<<"OURS PROGRAM ALLOCATED"<<endl;
        begin = chrono::steady_clock::now();
        populate_mram_parallel(system, subgraph);
        end = chrono::steady_clock::now();
        time_ours.transfer = chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9;
        // cout<<"DATA TRANSFER TIME: "<<chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9 <<" secs"<<endl;
        uint32_t num_v = subgraph.dpu_param[0][0].num_v_origin;
        uint32_t comp_size = ROUND_UP_TO_MULTIPLE_OF_2(num_v);
        bool check = true;
        while (check) {
            check = false;
            copy_comp_to_dpu(system, subgraph);

            begin = chrono::steady_clock::now();
            auto system_async = system.async();
            system_async.call(run_async);
            system_async.sync();
            end = chrono::steady_clock::now();
            time_ours.run = chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9;
            // cout<<"HOST ELAPSED TIME: "<<chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9 <<" secs."<<endl;
            for (uint32_t i = 0; i < NR_DPUS; i++) {
                auto dpu = system.dpus()[i];
                dpu->log(cout);
            }

            begin = chrono::steady_clock::now();

            /////////////////////////////////////////////////////////////////////////
            // This block is needed to optimize
            // Copy components
            vector<vector<uint32_t>> comp_temp;
            for (uint32_t i = 0; i < NR_DPUS; i++) {
                auto dpu = system.dpus()[i];
                if (i == 0)
                    dpu->copy(subgraph.comp, static_cast<unsigned>(comp_size * sizeof(uint32_t)), DPU_MRAM_HEAP_POINTER_NAME, subgraph.dpu_param[i][0].comp_start);
                else {
                    dpu->copy(comp_temp, static_cast<unsigned>(comp_size * sizeof(uint32_t)), DPU_MRAM_HEAP_POINTER_NAME, subgraph.dpu_param[i][0].comp_start);
                    for (uint32_t j = 0; j < num_v; j++) {
                        if (subgraph.comp[0][j] > comp_temp[0][j])
                            subgraph.comp[0][j] = comp_temp[0][j];
                    }
                }
            }

            #pragma omp parallel for
            for (uint32_t i = 0; i < num_v; i++) {
                while (subgraph.comp[0][i] != subgraph.comp[0][subgraph.comp[0][i]])
                    subgraph.comp[0][i] = subgraph.comp[0][subgraph.comp[0][i]];
            }
            /////////////////////////////////////////////////////////////////////////

            system.copy(result, static_cast<unsigned>(sizeof(uint64_t)), DPU_MRAM_HEAP_POINTER_NAME, subgraph.dpu_param[0][0].flag_start);
            end = chrono::steady_clock::now();
            time_ours.output_return = chrono::duration_cast<chrono::nanoseconds>(end - begin).count() / 1.0e9;

            for (uint32_t i = 0; i < NR_DPUS; i++) {
                if (result[i][0] == 1) {
                    check = true;
                    break;
                }
            }

            cout<<"OUTPUT RECEIVED"<<endl;
        }
        
        time_ours.total = time_ours.transfer + time_ours.run + time_ours.output_return;
        cout<<"PROGRAM END"<<endl<<endl;

    } catch (const DpuError & e) {
        cerr << e.what() << endl;
    }

    return 0;
}

void print_time(TimeRecord& time_base, TimeRecord& time_ours) {
    // cout<<"BASELINE TIME RESULT"<<endl;
    // cout<<"-------------------------------"<<endl;
    // cout<<"TRANSFER TIME: "<<time_base.transfer<<endl;
    // cout<<"DPU TIME: "<<time_base.run<<endl;
    // cout<<"OUTPUT RECEIVED TIME: "<<time_base.output_return<<endl<<endl;
    // cout<<"BASELINE TOTAL TIME: "<<time_base.total<<" sec"<<endl;
    // cout<<"-------------------------------"<<endl<<endl;

    cout<<"OURS TIME RESULT"<<endl;
    cout<<"-------------------------------"<<endl;
    cout<<"TRANSFER TIME: "<<time_ours.transfer<<endl;
    cout<<"DPU TIME: "<<time_ours.run<<endl;
    cout<<"OUTPUT RECEIVED TIME: "<<time_ours.output_return<<endl<<endl;
    cout<<"OURS TOTAL TIME: "<<time_ours.total<<" sec"<<endl;
    cout<<"-------------------------------"<<endl<<endl;

    // cout<<"SPEED UP (BASELINE / OURS): "<<time_base.total / time_ours.total<<endl;
}
