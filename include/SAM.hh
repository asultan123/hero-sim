#if !defined(__SAM_CPP__)
#define __SAM_CPP__

#include "AddressGenerator.hh"
#include "GlobalControl.hh"
#include <assert.h>
#include <iostream>
#include <string>
#include <systemc.h>

using std::cout;
using std::endl;
using std::string;
using namespace sc_core;
using namespace sc_dt;

template <typename DataType> struct SAMDataPortCreator
{
    SAMDataPortCreator(unsigned int _width, sc_trace_file *_tf);
    sc_vector<DataType> *operator()(const char *name, size_t);
    sc_trace_file *tf;
    unsigned int width;
};

template <typename DataType> using OutDataPortCreator = SAMDataPortCreator<sc_out<DataType>>;

template <typename DataType> using InDataPortCreator = SAMDataPortCreator<sc_in<DataType>>;

template <typename DataType> struct SAM : public sc_module
{
    // Member Signals
  private:
    sc_in_clk _clk;

  public:
    sc_port<GlobalControlChannel_IF> control;
    Memory<DataType> mem;
    sc_vector<AddressGenerator<DataType>> generators;
    sc_vector<MemoryChannel<DataType>> channels;
    sc_vector<sc_vector<sc_out<DataType>>> read_channel_data;
    sc_vector<sc_vector<sc_in<DataType>>> write_channel_data;
    const unsigned int length, width, channel_count;

    void update();

    void in_port_propogate();

    void out_port_propogate();

    SAM(sc_module_name name, GlobalControlChannel &_control, unsigned int _channel_count, unsigned int _length,
        unsigned int _width, sc_trace_file *tf, bool _trace_mem = false);

    SC_HAS_PROCESS(SAM);
};

template <typename DataType>
SAMDataPortCreator<DataType>::SAMDataPortCreator(unsigned int _width, sc_trace_file *_tf) : tf(_tf), width(_width)
{
}

template <typename DataType> sc_vector<DataType> *SAMDataPortCreator<DataType>::operator()(const char *name, size_t)
{
    return new sc_vector<DataType>(name, width);
}

template <typename DataType> void SAM<DataType>::update()
{
    if (control->reset())
    {
    }
    else if (control->enable())
    {
    }
}

template <typename DataType> void SAM<DataType>::in_port_propogate()
{
    if (control->enable())
    {
        for (unsigned int channel_index = 0; channel_index < channel_count; channel_index++)
        {
            if (channels[channel_index].mode() == MemoryChannelMode::WRITE)
            {
                for (unsigned int bus_index = 0; bus_index < width; bus_index++)
                {
                    channels[channel_index].channel_write_data_element(
                        write_channel_data[channel_index][bus_index].read(), bus_index);
                }
            }
        }
    }
}

template <typename DataType> void SAM<DataType>::out_port_propogate()
{
    if (control->enable())
    {
        for (unsigned int channel_index = 0; channel_index < channel_count; channel_index++)
        {
            if (channels[channel_index].mode() == MemoryChannelMode::READ)
            {
                for (unsigned int bus_index = 0; bus_index < width; bus_index++)
                {
                    read_channel_data[channel_index][bus_index] =
                        channels[channel_index].get_channel_read_data_bus()[bus_index];
                }
            }
        }
    }
}

// Constructor
template <typename DataType>
SAM<DataType>::SAM(sc_module_name name, GlobalControlChannel &_control, unsigned int _channel_count,
                   unsigned int _length, unsigned int _width, sc_trace_file *tf, bool _trace_mem)
    : sc_module(name), mem("mem", _control, _channel_count, _length, _width, tf, _trace_mem),
      generators("generator", _channel_count, AddressGeneratorCreator<DataType>(_control, tf)),
      channels("channels", _channel_count, MemoryChannelCreator<DataType>(_width, tf)),
      read_channel_data("read_channel_data", _channel_count, OutDataPortCreator<DataType>(_width, tf)),
      write_channel_data("write_channel_data", _channel_count, InDataPortCreator<DataType>(_width, tf)),
      length(_length), width(_width), channel_count(_channel_count)
{
    control(_control);
    _clk(control->clk());

    SC_METHOD(update);
    sensitive << _clk.pos();
    sensitive << control->reset();

    SC_METHOD(in_port_propogate);

    for (unsigned int channel_index = 0; channel_index < channel_count; channel_index++)
    {
        for (unsigned int data_index = 0; data_index < width; data_index++)
        {
            sensitive << write_channel_data[channel_index][data_index];
#ifdef DEBUG
            sc_trace(tf, write_channel_data[channel_index][data_index],
                     write_channel_data[channel_index][data_index].name());
#endif // DEBUG
        }
    }

    SC_METHOD(out_port_propogate)

    for (unsigned int channel_index = 0; channel_index < channel_count; channel_index++)
    {
        for (unsigned int data_index = 0; data_index < width; data_index++)
        {
            sensitive << channels[channel_index].get_channel_read_data_bus()[data_index];
#ifdef DEBUG
            sc_trace(tf, read_channel_data[channel_index][data_index],
                     read_channel_data[channel_index][data_index].name());
#endif // DEBUG
        }
    }

    for (unsigned int channel_index = 0; channel_index < channel_count; channel_index++)
    {
        generators[channel_index].channel(channels.at(channel_index));
        mem.channels[channel_index](channels.at(channel_index));
    }

#ifdef DEBUG
    cout << " SAM MODULE: " << name << " has been instantiated " << endl;
#endif // DEBUG
}

#endif
