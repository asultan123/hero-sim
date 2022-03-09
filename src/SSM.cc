#ifdef __INTELLISENSE__
#include "../include/SSM.hh"
#endif

template <typename DataType>
SSM<DataType>::SSM(sc_module_name name, GlobalControlChannel &_control, unsigned int _input_count,
                   unsigned int _output_count, sc_trace_file *_tf)
    : sc_module(name), input_count(_input_count), output_count(_output_count), in("in", _input_count),
      out("out", _output_count), in_generators("generator"), out_generators("generator"), in_channels("channels"),
      out_channels("channels")
{
    SC_METHOD(update);
    for (const auto &port : in)
    {
        sensitive << port;
    }

    if (input_count <= 0 || output_count <= 0)
    {
        throw std::invalid_argument("SSM Instantiation requires positive in/out port counts");
    }

    if (input_count > 1)
    {
        in_channels.init(input_count, MemoryChannelCreator<DataType>(1, _tf));
        in_generators.init(input_count, AddressGeneratorCreator<DataType>(_control, _tf));
        sc_assemble_vector(in_generators, &AddressGenerator<DataType>::channel).bind(in_channels);
    }
    if (output_count > 1)
    {
        out_channels.init(output_count, MemoryChannelCreator<DataType>(1, _tf));
        out_generators.init(output_count, AddressGeneratorCreator<DataType>(_control, _tf));
        sc_assemble_vector(in_generators, &AddressGenerator<DataType>::channel).bind(in_channels);
    }

    cout << "SSM MODULE: " << name << " has been instantiated " << endl;
}

template <typename DataType> void SSM<DataType>::update()
{
}