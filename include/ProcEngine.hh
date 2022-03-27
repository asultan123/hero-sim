#if !defined(__PROCENGINE_CC__)
#define __PROCENGINE_CC__

#include "AddressGenerator.hh"
#include "GlobalControl.hh"
#include <assert.h>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <systemc.h>
#include <vector>

using std::cout;
using std::endl;
using std::string;
using std::vector;
using namespace sc_core;
using namespace sc_dt;

template <typename DataType> struct PE : public sc_module
{

  public:
    sc_trace_file *tf;
    vector<int> weights;
    int weight_idx;
    sc_signal<DataType> psum_in;
    sc_signal<int> current_weight;
    int prog_idx;
    bool programmed;
    vector<Descriptor_2D> program;
    int weight_access_counter;
    int active_counter;
    int inactive_counter;

    void reset();

    DataType compute(sc_signal<DataType> &input);
    DataType compute(unsigned long int input);

    void resetWeightIdx();

    void resetWeights();

    void loadWeights(vector<int> &weights);

    void updateState();

    void loadProgram(vector<Descriptor_2D> &_program);

    // Constructor
    PE(sc_module_name name, sc_trace_file *_tf);
};

#include "../src/ProcEngine.cc"

#endif
