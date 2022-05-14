#if !defined(__GLOBAL_CONTROL_CPP__)
#define __GLOBAL_CONTROL_CPP__

#include <systemc.h>

using namespace sc_core;
using namespace sc_dt;

struct GlobalControlChannel_IF : virtual public sc_interface
{
  public:
    virtual sc_clock &clk() = 0;
    virtual const sc_signal<bool> &reset() = 0;
    virtual const sc_signal<bool> &program() = 0;
    virtual const sc_signal<bool> &enable() = 0;
    virtual void set_program(bool val) = 0;
    virtual void set_reset(bool val) = 0;
    virtual void set_enable(bool val) = 0;
};

struct GlobalControlChannel : public sc_module, public GlobalControlChannel_IF
{
    sc_clock global_clock;
    sc_signal<bool> global_reset;
    sc_signal<bool> global_enable;
    sc_signal<bool> global_program;
    GlobalControlChannel(sc_module_name name, sc_time time_val, sc_trace_file *tf);

    sc_clock &clk();

    sc_signal<bool> &reset();

    sc_signal<bool> &program();

    sc_signal<bool> &enable();

    void set_program(bool val);

    void set_reset(bool val);

    void set_enable(bool val);
};

// Not a template library but for the sake of consistency

GlobalControlChannel::GlobalControlChannel(sc_module_name name, sc_time time_val, sc_trace_file *tf)
    : sc_module(name), global_clock("clock", time_val), global_reset("reset"), global_enable("enable"),
      global_program("program")
{
#ifdef DEBUG
    sc_trace(tf, this->global_clock, (this->global_clock.name()));
    sc_trace(tf, this->global_reset, (this->global_reset.name()));
    sc_trace(tf, this->global_program, (this->global_program.name()));
    sc_trace(tf, this->global_enable, (this->global_enable.name()));
#endif // DEBUG
}

sc_clock &GlobalControlChannel::clk()
{
    return global_clock;
}

sc_signal<bool> &GlobalControlChannel::reset()
{
    return global_reset;
}

sc_signal<bool> &GlobalControlChannel::program()
{
    return global_program;
}
sc_signal<bool> &GlobalControlChannel::enable()
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

#endif