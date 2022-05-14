#if !defined(__DESCRIPTOR_HH__)
#define __DESCRIPTOR_HH__

#include "GlobalControl.hh"
#include "Memory.hh"
#include "memory.h"
#include <assert.h>
#include <iostream>
#include <map>
#include <string>
#include <systemc.h>
#include <type_traits>
#include <vector>

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
    WAIT,      // do nothing for certain number of cycles
    GENHOLD,
    RGENWAIT
};

struct Descriptor_2D
{
    unsigned int next;     // index of next descriptor
    unsigned int start;    // start index in ram array
    DescriptorState state; // state of dma
    unsigned int repeat;   // number of floats to transfer/wait
    unsigned int x_count;  // number of floats to transfer/wait
    int x_modify;          // number of floats between each transfer/wait
    unsigned int y_count;  // number of floats to transfer/wait
    int y_modify;          // number of floats between each transfer/wait
    int x_counter;
    int y_counter;

    Descriptor_2D(unsigned int _next, unsigned int _start, DescriptorState _state, unsigned int _x_count, int _x_modify,
                  unsigned int _y_count, int _y_modify);

    Descriptor_2D(const Descriptor_2D &rhs);

    bool operator==(const Descriptor_2D &rhs);

    void x_count_update(int count);

    void y_count_update(int count);

    template <typename ProgramContainer> static void make_sequential(ProgramContainer &program);

    static Descriptor_2D delay_inst(int delay_time);
    static Descriptor_2D delay_inst(int start, int delay_time);
    static Descriptor_2D stream_inst(int start_idx, int stream_size, int repeats);
    static Descriptor_2D genhold_inst(int start_idx, int hold_time, int repeats, int access_offset);
    static Descriptor_2D suspend_inst();

    static Descriptor_2D default_descriptor();
};

Descriptor_2D::Descriptor_2D(unsigned int _next, unsigned int _start, DescriptorState _state, unsigned int _x_count,
                             int _x_modify, unsigned int _y_count, int _y_modify)
{
    this->next = _next;
    this->start = _start;
    this->state = _state;
    this->x_count = _x_count;
    this->x_modify = _x_modify;
    this->y_count = _y_count;
    this->y_modify = _y_modify;
    this->x_counter = _x_count;
    this->y_counter = _y_count;
    this->repeat = 0;
}

Descriptor_2D::Descriptor_2D(const Descriptor_2D &rhs)
{
    this->next = rhs.next;
    this->start = rhs.start;
    this->state = rhs.state;
    this->x_count = rhs.x_count;
    this->x_modify = rhs.x_modify;
    this->y_count = rhs.y_count;
    this->y_modify = rhs.y_modify;
    this->x_counter = rhs.x_counter;
    this->y_counter = rhs.y_counter;
    this->repeat = rhs.repeat;
}

Descriptor_2D Descriptor_2D::default_descriptor()
{
    return {0, 0, DescriptorState::SUSPENDED, 0, 0, 0, 0};
}

template <typename ProgramContainer> void Descriptor_2D::make_sequential(ProgramContainer &program)
{
    static_assert(std::is_same<typename ProgramContainer::value_type, Descriptor_2D>::value,
                  "Program container must contain only Descriptor_2Ds");

    int idx = 1;
    for (auto &desc : program)
    {
        desc.next = idx++;
    }
    idx -= 2;
    program.at(program.size() - 1).next = idx;
}

Descriptor_2D Descriptor_2D::delay_inst(int delay_time)
{
    return Descriptor_2D(
        /*next*/ 0,
        /*start*/ 0,
        /*state*/ DescriptorState::WAIT,
        /*x_count*/ delay_time,
        /*x_modify*/ 0,
        /*y_count*/ 0,
        /*y_modify*/ 0);
}

Descriptor_2D Descriptor_2D::delay_inst(int start, int delay_time)
{
    return Descriptor_2D(
        /*next*/ 0,
        /*start*/ start,
        /*state*/ DescriptorState::WAIT,
        /*x_count*/ delay_time,
        /*x_modify*/ 0,
        /*y_count*/ 0,
        /*y_modify*/ 0);
}

Descriptor_2D Descriptor_2D::stream_inst(int start_idx, int stream_size, int repeats)
{
    return Descriptor_2D(
        /*next*/ 0,
        /*start*/ start_idx,
        /*state*/ DescriptorState::GENERATE,
        /*x_count*/ stream_size,
        /*x_modify*/ 1,
        /*y_count*/ repeats,
        /*y_modify*/ -(stream_size));
}

Descriptor_2D Descriptor_2D::genhold_inst(int start_idx, int hold_time, int repeats, int access_offset)
{
    return Descriptor_2D(
        /*next*/ 0,
        /*start*/ start_idx,
        /*state*/ DescriptorState::GENHOLD,
        /*x_count*/ hold_time,
        /*x_modify*/ 1,
        /*y_count*/ repeats,
        /*y_modify*/ access_offset);
}

Descriptor_2D Descriptor_2D::suspend_inst()
{
    return {0, 0, DescriptorState::SUSPENDED, 0, 0, 0, 0};
}

void Descriptor_2D::x_count_update(int count)
{
    this->x_count = count;
    this->x_counter = count;
}

void Descriptor_2D::y_count_update(int count)
{
    this->y_count = count;
    this->y_counter = count;
}

bool Descriptor_2D::operator==(const Descriptor_2D &rhs)
{
    return this->next == rhs.next && this->start == rhs.start && this->state == rhs.state &&
           this->x_count == rhs.x_count && this->x_modify == rhs.x_modify && this->y_count == rhs.y_count &&
           this->y_modify == rhs.y_modify;
}

#endif