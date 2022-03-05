#ifndef __SSM_HH__
#define __SSM_HH__

#include <assert.h>
#include <systemc.h>

#include <iostream>
#include <string>

#include "GlobalControl.hh"

using std::cout;
using std::endl;
using std::string;

// TODO: #1
template <typename DataType> struct SSM : public sc_module
{
    // Member Signals
  private:
    sc_in_clk _clk;

  public:
    sc_port<GlobalControlChannel_IF> control;
    unsigned int input_count;
    unsigned int output_count;
    sc_vector<sc_out<DataType>> in;
    sc_vector<sc_in<DataType>> out;

    void update();

    // Constructor
    SSM(sc_module_name name, GlobalControlChannel &_control, unsigned int input_count, unsigned int output_count,
        sc_trace_file *tf);

    SC_HAS_PROCESS(SSM);
};

#include "../src/SSM.cc"

#endif