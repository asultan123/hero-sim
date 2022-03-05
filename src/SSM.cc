#ifdef __INTELLISENSE__
#include "../include/SSM.hh"
#endif

template <typename DataType>
SSM<DataType>::SSM(sc_module_name name, GlobalControlChannel &_control, unsigned int _input_count,
                   unsigned int _output_count, sc_trace_file *tf)
    : sc_module(name), input_count(_input_count), output_count(_output_count), in("in", _input_count),
      out("out", _output_count)
{
    control(_control);
    _clk(control->clk());
    SC_METHOD(update);
    sensitive << _clk.pos();
    sensitive << control->reset();
    cout << "SSM MODULE: " << name << " has been instantiated " << endl;
}

template <typename DataType> void SSM<DataType> : update()
{
    if (control->reset())
    {
    }
    else if (control->enable())
    {
    }
}