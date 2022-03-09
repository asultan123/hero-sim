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

// TODO: #1
template <typename DataType> struct SSM : public sc_module
{
  public:
    unsigned int input_count;
    unsigned int output_count;
    sc_vector<sc_in<DataType>> in;
    sc_vector<sc_signal<DataType>> in_sig;
    sc_vector<sc_out<DataType>> out;
    sc_vector<sc_signal<DataType>> out_sig;
    std::unique_ptr<AddressGenerator<DataType>> in_generator;
    std::unique_ptr<AddressGenerator<DataType>> out_generator;
    std::unique_ptr<MemoryChannel<DataType>> in_channel;
    std::unique_ptr<MemoryChannel<DataType>> out_channel;

    void propogate_in_to_out();
    void load_in_port_program(const vector<Descriptor_2D> &newProgram);
    void load_out_port_program(const vector<Descriptor_2D> &newProgram);

    // Constructor
    SSM(sc_module_name name, GlobalControlChannel &_control, unsigned int input_count, unsigned int output_count,
        sc_trace_file *tf);

    SC_HAS_PROCESS(SSM);
};

#include "../src/SSM.cc"

#endif