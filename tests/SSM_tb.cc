#include "SSM.hh"
#include <fmt/format.h>
#include <random>
#include <systemc.h>
// #define DEBUG
using std::cout;
using std::endl;

#define SEED 1234
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
        fmt::print("Instantiated SSM TB with name {}\n", this->name());
    }
    bool validate_multi_in_single_out()
    {
        control.set_reset(true);
        control.set_program(false);
        sc_start(2, SC_NS);
        control.set_reset(false);
        control.set_program(true);

        std::default_random_engine eng{SEED};
        std::uniform_real_distribution<float> inst_dist(16, 32);
        auto inst_count = static_cast<int>(inst_dist(eng));

        std::uniform_real_distribution<float> port_dist(0, 16);
        std::uniform_real_distribution<float> wait_dist(16, 128);

        std::vector<Descriptor_2D> mux_program;
        for (int inst = 0; inst < inst_count; inst++)
        {
            auto selected_port = static_cast<int>(port_dist(eng));
            auto wait_time = static_cast<int>(wait_dist(eng));
            mux_program.push_back(Descriptor_2D::delay_inst(selected_port, wait_time));
        }

        Descriptor_2D::make_sequential(mux_program);

        dut.load_in_port_program(mux_program);

        sc_start(1, SC_NS);
        control.set_reset(false);
        control.set_program(false);
        control.set_enable(true);

        return true;
    }
    int run_tb()
    {
        cout << "Validating Reset" << endl;
        if (!validate_multi_in_single_out())
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