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
    sc_vector<sc_signal<DataType>> in_sig;
    sc_vector<sc_signal<DataType>> out_sig;

    SSM_TB(sc_module_name name)
        : sc_module(name), tf(sc_create_vcd_trace_file("SSM_TB")),
          control("global_control_channel", sc_time(1, SC_NS), tf), dut("dut", control, input_size, output_size, tf),
          in_sig("in_sig", input_size), out_sig("out_sig", output_size)
    {
        dut.in.bind(in_sig);
        dut.out.bind(out_sig);
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

        auto inst_count = 1024;

        std::default_random_engine eng{SEED};
        std::uniform_real_distribution<float> port_dist(0, 16);
        std::uniform_real_distribution<float> wait_dist(16, 128);
        std::uniform_real_distribution<float> data_dist(0, 255);

        std::vector<Descriptor_2D> mux_program;
        for (int inst = 0; inst < inst_count; inst++)
        {
            auto selected_port = static_cast<int>(port_dist(eng));
            auto wait_time = static_cast<int>(wait_dist(eng));
            mux_program.push_back(Descriptor_2D::delay_inst(selected_port, wait_time));
        }
        mux_program.push_back(Descriptor_2D::suspend_inst());
        Descriptor_2D::make_sequential(mux_program);

        dut.load_in_port_program(mux_program);

        sc_start(1, SC_NS);
        control.set_reset(false);
        control.set_program(false);
        control.set_enable(true);

        for (const auto &descriptor : mux_program)
        {
            if (descriptor.state == DescriptorState::SUSPENDED)
            {
                break;
            }
            int descriptor_time = descriptor.x_count + 1;
            int current_port = descriptor.start;
            for (int t = 0; t < descriptor_time; t++)
            {
                // write new data to all mux ports
                for (unsigned int port_idx = 0; port_idx < dut.in.size(); port_idx++)
                {
                    auto rand_data = static_cast<DataType>(data_dist(eng));
                    in_sig.at(port_idx).write(rand_data);
                }

                // allow inputs/outputs to update
                sc_start(1, SC_NS);

                // Validate out port and in port
                if (out_sig.at(0).read() != in_sig.at(current_port).read())
                {
                    auto current_time = sc_time_stamp().to_default_time_units();
                    fmt::print("Invalid value for mux out port at time {}\n", current_time);
                    fmt::print("Value @out_sig {}\nValue @in_sig {}\n", out_sig.at(0).read(),
                               in_sig.at(current_port).read());
                    sc_start(10, SC_NS);
                    return false;
                }
            }
            sc_start(1, SC_NS); // allow transition to next descriptor
        }

        return true;
    }
    int run_tb()
    {
        cout << "Validating Reset" << endl;
        if (!validate_multi_in_single_out())
        {
            cout << "validate_multi_in_single_out" << endl;
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