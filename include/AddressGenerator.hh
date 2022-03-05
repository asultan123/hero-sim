#if !defined(__ADDRESS_GENERATOR_HH__)
#define __ADDRESS_GENERATOR_HH__

#include "Descriptor.hh"
#include "GlobalControl.hh"
#include "Memory.hh"
#include "memory.h"
#include <assert.h>
#include <iostream>
#include <map>
#include <string>
#include <systemc>
#include <vector>

using std::cout;
using std::endl;
using std::string;
using std::vector;
using namespace sc_core;
using namespace sc_dt;

template <typename DataType> struct AddressGenerator : public sc_module
{
    // Control Signals
  private:
    sc_in_clk _clk;
    sc_in<bool> _reset;

  public:
    sc_port<GlobalControlChannel_IF> control;
    sc_port<MemoryChannel_IF<DataType>> channel;
    sc_trace_file *tf;

    // Internal Data
    vector<Descriptor_2D> descriptors;
    sc_signal<unsigned int> execute_index;
    sc_signal<unsigned int> current_ram_index;
    sc_signal<unsigned int> x_count_remaining;
    sc_signal<unsigned int> y_count_remaining;
    sc_signal<unsigned int> repeat;
    sc_signal<bool> programmed;
    sc_signal<bool> first_cycle;
    sc_signal<bool> last_cycle;

    void resetIndexingCounters();

    void loadInternalCountersFromIndex(unsigned int index);

    void loadProgram(const vector<Descriptor_2D> &newProgram);

    void resetProgramMemory();

    Descriptor_2D currentDescriptor();
    Descriptor_2D nextDescriptor();

    void updateCurrentIndex();

    void RGENWAITupdateCurrentIndex();

    bool descriptorComplete();

    void loadNextDescriptor();

    void update();

    // Constructor
    AddressGenerator(sc_module_name name, GlobalControlChannel &_control, sc_trace_file *_tf);

    SC_HAS_PROCESS(AddressGenerator);
};

template <typename DataType> struct AddressGeneratorCreator
{
    AddressGeneratorCreator(GlobalControlChannel &_control, sc_trace_file *_tf);
    AddressGenerator<DataType> *operator()(const char *name, size_t);
    sc_trace_file *tf;
    GlobalControlChannel &control;
};

#include "../src/AddressGenerator.cc"

#endif
