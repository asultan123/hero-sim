#if !defined(__PE__CC__)
#define __PE__CC__

#include <systemc>
#include <map>
#include <vector>
#include "GlobalControl.hh"
#include "AddressGenerator.hh"
#include <assert.h>
#include <iostream>
#include <string>
#include <stdexcept>

using std::cout;
using std::endl;
using std::string;
using std::vector;
using namespace sc_core;
using namespace sc_dt;

template <typename DataType>
struct PE : public sc_module
{

public:

    sc_trace_file* tf;
    vector<int> weights;
    int weight_idx;
    sc_signal<DataType> psum_in;
    int prog_idx;
    bool programmed;
    vector<Descriptor_2D> program;

    void reset();

    int currentWeight();

    void resetWeightIdx();

    void resetWeights();

    void loadWeights(vector<int>& weights);

    void updateState();

    void loadProgram(vector<Descriptor_2D>& _program);

    // Constructor
    PE(sc_module_name name, sc_trace_file* _tf);
};

#endif
