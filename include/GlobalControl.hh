#if !defined(__GLOBAL_CONTROL_CPP__)
#define __GLOBAL_CONTROL_CPP__

#include <systemc>

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
    GlobalControlChannel(sc_module_name name,
                         sc_time time_val,
                         sc_trace_file *tf);

    sc_clock &clk();
    
    sc_signal<bool> &reset();

    sc_signal<bool> &program();

    sc_signal<bool> &enable();

    void set_program(bool val);

    void set_reset(bool val);

    void set_enable(bool val);
};

// Not a template library but for the sake of consistency
#include "../src/GlobalControl.cc"

#endif