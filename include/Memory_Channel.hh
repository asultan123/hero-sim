#if !defined(__MEMORY_CHANNEL_CPP__)
#define __MEMORY_CHANNEL_CPP__

#include "GlobalControl.hh"
#include <assert.h>
#include <iostream>
#include <string>
#include <systemc>

using namespace sc_core;
using namespace sc_dt;
using std::cout;
using std::endl;
using std::string;

enum MemoryChannelMode
{
    READ = 1,
    WRITE = 2
};

template <typename DataType> struct MemoryChannel_IF : virtual public sc_interface
{
  public:
    // control
    virtual void reset() = 0;
    virtual void set_enable(bool status) = 0;
    virtual void set_addr(unsigned int addr) = 0;
    virtual void set_mode(MemoryChannelMode mode) = 0;

    // Data
    virtual MemoryChannelMode mode() = 0;
    virtual const sc_vector<sc_signal<DataType, SC_MANY_WRITERS>> &mem_read_data() = 0;
    virtual void mem_write_data(const sc_vector<sc_signal<DataType>> &_data) = 0;
    virtual void mem_write_data(int _data) = 0;
    virtual const sc_vector<sc_signal<DataType, SC_MANY_WRITERS>> &channel_read_data() = 0;
    virtual const DataType &channel_read_data_element(unsigned int col) = 0;
    virtual sc_vector<sc_signal<DataType, SC_MANY_WRITERS>> &get_channel_read_data_bus() = 0;
    virtual sc_vector<sc_signal<DataType, SC_MANY_WRITERS>> &get_channel_write_data_bus() = 0;
    virtual void channel_write_data(const sc_vector<sc_signal<DataType>> &_data) = 0;
    virtual void channel_write_data_element(DataType _data, unsigned int col) = 0;
    virtual unsigned int addr() = 0;
    virtual bool enabled() = 0;
    virtual const unsigned int &get_width() = 0;
};

template <typename DataType> struct MemoryChannel : public sc_channel, public MemoryChannel_IF<DataType>
{
    sc_vector<sc_signal<DataType, SC_MANY_WRITERS>> read_channel_data;
    sc_vector<sc_signal<DataType, SC_MANY_WRITERS>> write_channel_data;
    sc_signal<unsigned int> channel_addr;
    sc_signal<bool> channel_enabled;
    sc_signal<unsigned int> channel_mode;
    const unsigned int channel_width;

    MemoryChannel(sc_module_name name, unsigned int width, sc_trace_file *tf);

    const sc_vector<sc_signal<DataType, SC_MANY_WRITERS>>
        &mem_read_data(); // TODO: #30 Rename confusing access function for Memory objects

    void mem_write_data(const sc_vector<sc_signal<DataType>> &_data);
    void mem_write_data(int _data); // TODO: #29 Remove ability to write int data to a memory channel

    const sc_vector<sc_signal<DataType, SC_MANY_WRITERS>> &channel_read_data();

    const DataType &channel_read_data_element(unsigned int col);

    sc_vector<sc_signal<DataType, SC_MANY_WRITERS>> &get_channel_read_data_bus();

    sc_vector<sc_signal<DataType, SC_MANY_WRITERS>> &get_channel_write_data_bus();

    void channel_write_data(const sc_vector<sc_signal<DataType>> &_data);

    void channel_write_data_element(DataType _data, unsigned int col);

    unsigned int addr();

    void set_addr(unsigned int addr);

    void set_enable(bool status);

    bool enabled();

    void set_mode(MemoryChannelMode mode);

    MemoryChannelMode mode();

    void reset();

    const unsigned int &get_width();

    void register_port(sc_port_base &port_, const char *if_typename_);
};

template <typename DataType> struct MemoryChannelCreator
{
    MemoryChannelCreator(unsigned int _width, sc_trace_file *_tf);
    MemoryChannel<DataType> *operator()(const char *name, size_t);
    sc_trace_file *tf;
    unsigned int width;
};

#include "../src/Memory_Channel.cc"

#endif