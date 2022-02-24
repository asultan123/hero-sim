#if !defined(__MEMORY_CPP__)
#define __MEMORY_CPP__

#include <assert.h>
#include <iostream>
#include <string>
#include <systemc>
#include "Memory_Channel.hh"
#include "GlobalControl.hh"

using std::cout;
using std::endl;
using std::string;
using namespace sc_core;
using namespace sc_dt;

template <typename DataType>
struct MemoryRowCreator
{
    MemoryRowCreator(unsigned int _width, sc_trace_file* _tf);
    sc_vector<sc_signal<DataType>>* operator()(const char* name, size_t);
    sc_trace_file* tf;
    unsigned int width;
};

template <typename DataType>
struct Memory : public sc_module
{
private:
    sc_in_clk _clk;
    // Control Signals
public:
    sc_vector<sc_vector<sc_signal<DataType>>> ram;
    sc_port<GlobalControlChannel_IF> control;
    sc_vector<sc_port<MemoryChannel_IF<DataType>>> channels;
    const unsigned int width, length, channel_count;
    int access_counter;

    void update();

    void print_memory_contents();

    // Constructor
    Memory(
        sc_module_name name,
        GlobalControlChannel& _control,
        unsigned int _channel_count,
        unsigned int _length,
        unsigned int _width,
        sc_trace_file* tf);

    SC_HAS_PROCESS(Memory);
};

#include "../src/Memory.cc"

#endif
