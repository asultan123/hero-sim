#ifdef __INTELLISENSE__
#include "../include/SSM.hh"
#endif

template <typename DataType>
SSM<DataType>::SSM(sc_module_name name, GlobalControlChannel &_control, unsigned int _input_count,
                   unsigned int _output_count, sc_trace_file *_tf)
    : sc_module(name), input_count(_input_count), output_count(_output_count), in("in"), in_sig("in_sig"), out("out"),
      out_sig("out_sig")
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
    in_sig.init(input_count);
    out.init(output_count);
    out_sig.init(output_count);

    in.bind(in_sig);
    out.bind(out_sig);

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
    cout << "SSM MODULE: " << name << " has been instantiated " << endl;
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
        auto target_port = out_channel->addr();
        out_sig.at(target_port).write(in_sig.at(0).read());
    }
    else if (input_count > 1)
    {
        if (in_channel == nullptr)
        {
            throw std::runtime_error(
                "Something went wrong during SSM instantiation, \"in\" channel should be non null");
        }
        auto target_port = in_channel->addr();
        out_sig.at(0).write(in_sig.at(target_port).read());
    }
    else
    {
        throw std::runtime_error("Something went wrong in SSM instantiation, port counts for both in/out ports can't "
                                 "both be less than or equal to 1");
    }
}