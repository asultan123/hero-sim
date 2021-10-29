#include "GlobalControl.hh"
#include <systemc>

GlobalControlChannel::GlobalControlChannel(sc_module_name name,
                        sc_time time_val,
                        sc_trace_file *tf) : sc_module(name),
                                                    global_clock("clock", time_val),
                                                    global_reset("reset"),
                                                    global_enable("enable"),
                                                    global_program("program")
{
    sc_trace(tf, this->global_clock, (this->global_clock.name()));
    sc_trace(tf, this->global_reset, (this->global_reset.name()));
    sc_trace(tf, this->global_program, (this->global_program.name()));
    sc_trace(tf, this->global_enable, (this->global_enable.name()));
}

sc_clock& GlobalControlChannel::clk()
{
    return global_clock;
}

sc_signal<bool>& GlobalControlChannel::reset()
{
    return global_reset;
}

sc_signal<bool>& GlobalControlChannel::program()
{
    return global_program;
}
sc_signal<bool>& GlobalControlChannel::enable()
{
    return global_enable;
}

void GlobalControlChannel::set_program(bool val)
{
    global_program = val;
}
void GlobalControlChannel::set_reset(bool val)
{
    global_reset = val;
}

void GlobalControlChannel::set_enable(bool val)
{
    global_enable = val;
}
