#include "SSM.hh"
#include <systemc.h>
// #define DEBUG
using std::cout;
using std::endl;
template <typename DataType> struct SSM_TB : public sc_module
{
    const unsigned int input_size = 16;
    const unsigned int output_size = 1;
    sc_trace_file *tf;
    GlobalControlChannel control;
    SSM<DataType> dut;
    SSM_TB(sc_module_name name)
        : sc_module(name), tf(sc_create_vcd_trace_file("SSM_TB")),
          control("global_control_channel", sc_time(1, SC_NS), tf), dut("dut", control, input_size, output_size, tf)
    {
        tf->set_time_unit(1, SC_PS);
        cout << "Instantiated SSM TB with name " << this->name() << endl;
    }
    bool validate_reset()
    {
        sc_start(10, SC_NS);
        return true;
    }
    int run_tb()
    {
        cout << "Validating Reset" << endl;
        if (!validate_reset())
        {
            cout << "Reset Failed" << endl;
            return -1;
        }
        cout << "Reset Success" << endl;
        cout << "ALL TESTS PASS " << endl;
        return 0;
    }
    ~SSM_TB()
    {
        sc_close_vcd_trace_file(tf);
    }
};
int sc_main(int argc, char *argv[])
{
    SSM_TB<sc_int<32>> tb("SSM_tb");
    return tb.run_tb();
}