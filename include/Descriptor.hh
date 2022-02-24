#if !defined(__DESCRIPTOR_HH__)
#define __DESCRIPTOR_HH__

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
    WAIT,       // do nothing for certain number of cycles
    GENHOLD,
    RGENWAIT
};

struct Descriptor_2D
{
    unsigned int next;     // index of next descriptor
    unsigned int start;    // start index in ram array
    DescriptorState state; // state of dma
    unsigned int repeat;  // number of floats to transfer/wait
    unsigned int x_count;  // number of floats to transfer/wait
    int x_modify;          // number of floats between each transfer/wait
    unsigned int y_count;  // number of floats to transfer/wait
    int y_modify;          // number of floats between each transfer/wait
    int x_counter;
    int y_counter;

    Descriptor_2D(unsigned int _next, unsigned int _start, DescriptorState _state,
                  unsigned int _x_count, int _x_modify, unsigned int _y_count,
                  int _y_modify);

    Descriptor_2D(const Descriptor_2D& rhs);

    bool operator==(const Descriptor_2D& rhs);

    void x_count_update(int count);
    
    void y_count_update(int count);

    static void make_sequential(vector<Descriptor_2D>& program);

    static Descriptor_2D delay_inst(int delay_time);
    static Descriptor_2D stream_inst(int start_idx, int stream_size, int repeats);
    static Descriptor_2D genhold_inst(int start_idx, int hold_time, int repeats, int access_offset);
    static Descriptor_2D suspend_inst();

    static Descriptor_2D default_descriptor();
};

#include "../src/Descriptor.cc"

#endif