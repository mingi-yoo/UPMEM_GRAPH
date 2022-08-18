#include <dpu>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <limits>

using namespace dpu;
using namespace std;

#ifndef DPU_BASELINE
#define DPU_BASLINE "pr_baseline"
#endif

#ifndef DPU_OURS
#define DPU_OURS "pr_ours"
#endif

#define NB_OF_DPUS 64

#define NB_OF_VERTICES 10000
#define NB_OF_EDGES 1000000

vector<uint32_t> row_ptr;
vector<uint32_t> col_idx;
vector<float> value;

void read_csr(string csr_path) {
    ifstream csr(csr_path);
    string line, temp;

    if (csr.is_open()) {
        getline(csr, line);
        stringstream ss(line);
        while (getline(ss, temp, ' '))
            row_ptr.push_back(stoi(temp));
        ss.clear();
        getline(csr, line);
        ss.str(line);
        while (getline(ss, temp, ' '))
            col_idx.push_back(stoi(temp));
        ss.clear();
        getline(a_file, line);
        ss.str(line);
        while (getline(ss, temp, ' '))
            value.push_back(stof(temp));
        csr.close();
    }
    else
        throw invalid_argument("Cannot open graph");
}

void populate_mram(DpuSetOps& dpu) {
    vector<uint32_t> row_buffer(NB_OF_VERTICES, numeric_limits<uint32_t>::max())
    vector<uint32_t> col_buffer(NB_OF_EDGES, numeric_limits<uint32_t>::max())
    vector<uint32_t> val_buffer(NB_OF_EDGES, numeric_limits<float>::max())

    // copy data for passing to MRAM
    for (int i = 0; i < row_ptr.size(); i++) {
        row_buffer[i] = row_ptr[i];

    for (int i = 0; i < col_idx.size(); i++) {
        col_buffer[i] = col_idx[i];
        val_buffer[i] = value[i];
    }

    dpu.copy("row_ptr", row_buffer, static_cast<unsigned>(NB_OF_VERTICES));
    dpu.copy("col_idx", col_buffer, static_cast<unsigned>(NB_OF_EDGES));
    dpu.copy("value", val_buffer, static_cast<unsigned>(NB_OF_EDGES));
}

void populate_mram(DpuSetOps& dpu, uint32_t id) {
    // TO-DO
}

int main(int argc, char** argv) {
    // read graph file
    read_csr(argv[1]);

    try {
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