#include <dpu>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

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

void populate_mram_base(DpuSetOps& dpu) {
    // TO-DO
}

void populate_mram_ours(DpuSetOps& dpu, uint32_t id) {
    // TO-DO
}

int main(int argc, char** argv) {
    // read graph file
    read_csr(argv[1]);

    try {
        auto system = DpuSet::allocate(NB_OF_DPUS);
        auto dpu_baseline = system.dpus()[0];
        dpu_baseline->load(DPU_BASELINE)

    } catch (const DpuError & e) {
        cerr << e.what() << endl;
    }
}