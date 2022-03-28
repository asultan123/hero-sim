#ifndef __SSM_HH__
#define __SSM_HH__

#include <assert.h>
#include <systemc.h>

#include <iostream>
#include <string>

#include "AddressGenerator.hh"
#include "GlobalControl.hh"

using std::cout;
using std::endl;
using std::string;

enum SSMMode
{
    STATIC = 1,
    DYNAMIC = 2
};

// TODO: #1
template <typename DataType> struct SSM : public sc_module
{
  public:
    unsigned int input_count;
    unsigned int output_count;
    sc_vector<sc_in<DataType>> in;
    sc_vector<sc_out<DataType>> out;
    std::unique_ptr<AddressGenerator<DataType>> in_generator;
    std::unique_ptr<AddressGenerator<DataType>> out_generator;
    std::unique_ptr<MemoryChannel<DataType>> in_channel;
    std::unique_ptr<MemoryChannel<DataType>> out_channel;
    SSMMode mode;
    int static_input_port_select;
    int static_output_port_select;

    void propogate_in_to_out();
    void load_in_port_program(const vector<Descriptor_2D> &newProgram);
    void set_static_input_port_select(int _static_input_port_select);
    void set_static_output_port_select(int _static_output_port_select);
    void load_out_port_program(const vector<Descriptor_2D> &newProgram);

    // Constructor
    SSM(sc_module_name name, GlobalControlChannel &_control, unsigned int input_count, unsigned int output_count,
        sc_trace_file *tf, SSMMode _mode);

    SC_HAS_PROCESS(SSM);
};

#include "../src/SSM.cc"

#endif