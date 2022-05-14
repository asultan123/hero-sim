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

template <typename DataType>
SSM<DataType>::SSM(sc_module_name name, GlobalControlChannel &_control, unsigned int _input_count,
                   unsigned int _output_count, sc_trace_file *_tf, SSMMode _mode)
    : sc_module(name), input_count(_input_count), output_count(_output_count), in("in"), out("out"), mode(_mode),
      static_input_port_select(0), static_output_port_select(0)
{
    if (input_count <= 0 || output_count <= 0)
    {
        throw std::invalid_argument("SSM Instantiation requires positive in/out port counts");
    }

    if (input_count == 1 && output_count == 1)
    {
        throw std::invalid_argument("SSM instantiation with 1 input and 1 output port is a wire, which is pointless");
    }

    if (input_count > 1 && output_count > 1)
    {
        throw std::invalid_argument(
            "SSM Instantiation requires port count for either input or output to be greater than 1, but not both");
    }

    in.init(input_count);
    out.init(output_count);

#ifdef DEBUG
    for (auto &port : in)
    {
        sc_trace(_tf, port, port.name());
    }
    for (auto &port : out)
    {
        sc_trace(_tf, port, port.name());
    }
#endif // DEBUG

    if (input_count > 1)
    {
        in_channel = std::unique_ptr<MemoryChannel<DataType>>(new MemoryChannel<DataType>("in_channel", 1, _tf));
        in_generator =
            std::unique_ptr<AddressGenerator<DataType>>(new AddressGenerator<DataType>("in_ag", _control, _tf));
        in_generator->channel(*in_channel);
    }
    if (output_count > 1)
    {
        out_channel = std::unique_ptr<MemoryChannel<DataType>>(new MemoryChannel<DataType>("out_channel", 1, _tf));
        out_generator =
            std::unique_ptr<AddressGenerator<DataType>>(new AddressGenerator<DataType>("out_ag", _control, _tf));
        out_generator->channel(*out_channel);
    }

    SC_METHOD(propogate_in_to_out);
    for (const auto &port : in)
    {
        sensitive << port;
    }
    if (input_count > 1)
    {
        sensitive << in_channel->channel_addr;
    }
#ifdef DEBUG
    cout << "SSM MODULE: " << name << " has been instantiated " << endl;
#endif // DEBUG
}

template <typename DataType> void SSM<DataType>::set_static_input_port_select(int _static_input_port_select)
{
    static_input_port_select = _static_input_port_select;
}

template <typename DataType> void SSM<DataType>::set_static_output_port_select(int _static_output_port_select)
{
    static_output_port_select = _static_output_port_select;
}

template <typename DataType> void SSM<DataType>::load_in_port_program(const vector<Descriptor_2D> &newProgram)
{
    if (input_count == 1)
    {
        throw std::invalid_argument("SSM only has one input port, no address generator available to program");
    }
    if (in_generator != nullptr)
    {
        in_generator->loadProgram(newProgram);
    }
    else
    {
        throw std::runtime_error("Something went wrong during SSM instantiation, \"in\" generator should be non null");
    }
}

template <typename DataType> void SSM<DataType>::load_out_port_program(const vector<Descriptor_2D> &newProgram)
{
    if (output_count == 1)
    {
        throw std::invalid_argument("SSM only has one output port, no address generator available to program");
    }
    if (out_generator != nullptr)
    {
        out_generator->loadProgram(newProgram);
    }
    else
    {
        throw std::runtime_error("Something went wrong in SSM instantiation, \"out\" generator should be non null");
    }
}

template <typename DataType> void SSM<DataType>::propogate_in_to_out()
{
    if (output_count > 1)
    {
        if (out_channel == nullptr)
        {
            throw std::runtime_error(
                "Something went wrong during SSM instantiation, \"out\" channel should be non null");
        }
        switch (mode)
        {
        case SSMMode::DYNAMIC:
        {
            auto target_port = out_channel->addr();
            out.at(target_port).write(in.at(0).read());
            break;
        }
        case SSMMode::STATIC:
        {
            out.at(static_output_port_select).write(in.at(0).read());
            break;
        }
        default:
        {
            throw "Invalid SSM Mode selected";
            break;
        }
        }
    }
    else if (input_count > 1)
    {
        if (in_channel == nullptr)
        {
            throw std::runtime_error(
                "Something went wrong during SSM instantiation, \"in\" channel should be non null");
        }
        switch (mode)
        {
        case SSMMode::DYNAMIC:
        {
            auto target_port = in_channel->addr();
            out.at(0).write(in.at(target_port).read());
            break;
        }
        case SSMMode::STATIC:
        {
            out.at(0).write(in.at(static_input_port_select).read());
            break;
        }
        default:
        {
            throw "Invalid SSM Mode selected";
            break;
        }
        }
    }
    else
    {
        throw std::runtime_error("Something went wrong in SSM instantiation, port counts for both in/out ports can't "
                                 "both be 1");
    }
}

#endif