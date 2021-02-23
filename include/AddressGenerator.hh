#if !defined(__ADDRESS_GENERATOR_CPP__)
#define __ADDRESS_GENERATOR_CPP__

#include <systemc>
#include <map>
#include <vector>
#include "GlobalControl.hh"
#include "memory.h"
#include "Memory.hh"
#include <assert.h>
#include <iostream>
#include <string>

using std::cout;
using std::endl;
using std::string;
using std::vector;
using namespace sc_core;
using namespace sc_dt;

enum class DescriptorState
{
    SUSPENDED, // do nothing indefinitely
    GENERATE,  // transfer data
    WAIT       // do nothing for certain number of cycles
};

struct Descriptor_2D
{
    unsigned int next;     // index of next descriptor
    unsigned int start;    // start index in ram array
    DescriptorState state; // state of dma
    unsigned int x_count;  // number of floats to transfer/wait
    int x_modify;          // number of floats between each transfer/wait
    unsigned int y_count;  // number of floats to transfer/wait
    int y_modify;          // number of floats between each transfer/wait

    Descriptor_2D(unsigned int _next, unsigned int _start, DescriptorState _state,
                  unsigned int _x_count, int _x_modify, unsigned int _y_count,
                  int _y_modify);

    bool operator==(const Descriptor_2D& rhs);

    static Descriptor_2D default_descriptor();
};



template <typename DataType>
struct AddressGenerator : public sc_module
{
    // Control Signals
private:
    sc_in_clk _clk;
    sc_in<bool> _reset;

public:
    sc_port<GlobalControlChannel_IF> control;
    sc_port<MemoryChannel_IF<DataType>> channel;
    sc_trace_file* tf;

    // Internal Data
    vector<Descriptor_2D> descriptors;
    sc_signal<unsigned int> execute_index;
    sc_signal<unsigned int> current_ram_index;
    sc_signal<unsigned int> x_count_remaining;
    sc_signal<unsigned int> y_count_remaining;
    sc_signal<bool> programmed;
    sc_signal<bool> first_cycle;
    sc_signal<bool> last_cycle;

    void resetIndexingCounters();

    void loadInternalCountersFromIndex(unsigned int index);

    void loadProgram(const vector<Descriptor_2D>& newProgram);

    void resetProgramMemory();

    Descriptor_2D currentDescriptor();
    Descriptor_2D nextDescriptor();

    void updateCurrentIndex();

    bool descriptorComplete();

    void loadNextDescriptor();

    void update(); 

    // Constructor
    AddressGenerator(sc_module_name name, GlobalControlChannel& _control,
                     sc_trace_file* _tf);

    SC_HAS_PROCESS(AddressGenerator);
};

template <typename DataType>
struct AddressGeneratorCreator
{
    AddressGeneratorCreator(GlobalControlChannel& _control, sc_trace_file* _tf);
    AddressGenerator<DataType>* operator()(const char* name, size_t);
    sc_trace_file* tf;
    GlobalControlChannel& control;
};

#endif
