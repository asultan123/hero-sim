#include "AddressGenerator.hh"
#include <systemc>
// #define DEBUG
using std::cout;
using std::endl;

template <typename DataType> struct AddressGenerator_TB : public sc_module
{

    const unsigned int mem_channel_width = 4;

    sc_trace_file *tf;
    GlobalControlChannel control;

    AddressGenerator<DataType> dut;
    MemoryChannel<DataType> mem_channel;
    AddressGenerator_TB(sc_module_name name)
        : sc_module(name), tf(sc_create_vcd_trace_file("ProgTrace")),
          control("global_control_channel", sc_time(1, SC_NS), tf), dut("dut", control, tf),
          mem_channel("mem_channel", mem_channel_width, tf)
    {

        tf->set_time_unit(1, SC_PS);
        dut.channel(mem_channel);
        control.set_reset(false);
        cout << "Instantiated AddressGenerator TB with name " << this->name() << endl;
    }
    bool validate_reset()
    {
        cout << "Validating Reset" << endl;
        control.set_program(false);
        control.set_reset(false);
        control.set_reset(false);
        sc_start(1, SC_NS);
        control.set_reset(true);
        sc_start(1, SC_NS);
        if (!(dut.descriptors.at(0) == Descriptor_2D::default_descriptor()))
        {
            cout << "dut.descriptors.at(0) == default_descriptor FAILED!" << endl;
            return false;
        }
        if (!(dut.execute_index == 0))
        {
            cout << "dut.execute_index == 0 FAILED!" << endl;
            return false;
        }
        if (!(mem_channel.addr() == 0))
        {
            cout << "mem_channel.addr() == 0 FAILED!" << endl;
            return false;
        }
        if (!(dut.x_count_remaining == 0))
        {
            cout << "dut.x_count_remaining == 0 FAILED!" << endl;
            return false;
        }
        if (!(dut.y_count_remaining == 0))
        {
            cout << "dut.y_count_remaining == 0 FAILED!" << endl;
            return false;
        }
        control.set_reset(false);
        sc_start(1, SC_NS);
        cout << "validate_reset Success" << endl;
        return true;
    }

    bool validate_loadprogram_and_suspended_state()
    {
        cout << "Validating validate_loadprogram_and_suspended_state" << endl;

        Descriptor_2D suspend_descriptor(0, 0, DescriptorState::SUSPENDED, 0, 0, 0, 0);
        vector<Descriptor_2D> temp_program;
        temp_program.push_back(suspend_descriptor);
        dut.loadProgram(temp_program);
        control.set_program(true);
        sc_start(1, SC_NS);
        control.set_program(false);
        sc_start(10, SC_NS);

        if (!(dut.descriptors.at(0) == suspend_descriptor))
        {
            cout << "dut.descriptors.at(0) == suspend_descriptor FAILED!" << endl;
            return false;
        }
        if (!(dut.currentDescriptor() == suspend_descriptor))
        {
            cout << "dut.currentDescriptor() == suspend_descriptor FAILED!" << endl;
            return false;
        }
        if (!(dut.execute_index == 0))
        {
            cout << "dut.execute_index == 0 FAILED!" << endl;
            return false;
        }
        if (!(dut.x_count_remaining == 0))
        {
            cout << "dut.x_count_remaining == 0 FAILED!" << endl;
            return false;
        }
        if (!(dut.y_count_remaining == 0))
        {
            cout << "dut.y_count_remaining == 0 FAILED!" << endl;
            return false;
        }
        cout << "validate_loadprogram_and_suspended_state Success" << endl;
        return true;
    }

    bool validate_wait_and_descriptor_retirement()
    {
        cout << "Validating validate_wait" << endl;

        control.set_enable(false);
        control.set_reset(true);
        sc_start(1, SC_NS);
        control.set_reset(false);
        sc_start(1, SC_NS);
        Descriptor_2D wait_descriptor_1(1, 0, DescriptorState::WAIT, 2, 2, 0, 0);
        Descriptor_2D wait_descriptor_2(2, 0, DescriptorState::WAIT, 3, 3, 0, 0);
        Descriptor_2D suspend_descriptor(3, 0, DescriptorState::SUSPENDED, 0, 0, 0, 0);
        vector<Descriptor_2D> temp_program;
        temp_program.push_back(wait_descriptor_1);
        temp_program.push_back(wait_descriptor_2);
        temp_program.push_back(suspend_descriptor);
        dut.loadProgram(temp_program);
        control.set_program(true);
        // load program and start first descriptor
        sc_start(1, SC_NS);
        control.set_program(false);
        control.set_enable(true);
        // first cycle
        sc_start(1, SC_NS);

        cout << "validate initial condition at first cycle" << endl;
        if (!(dut.current_ram_index == 0))
        {
            cout << "dut.current_ram_index == 0 FAILED!" << endl;
            return false;
        }
        if (!(dut.x_count_remaining == 2 && dut.y_count_remaining == 0))
        {
            cout << "dut.x_count_remaining == 2 && dut.y_count_remaining == 0 FAILED!" << endl;
            return false;
        }
        cout << "validate success!" << endl;

        cout << "validate first wait descriptor execution" << endl;
        for (unsigned int i = 2; i <= 4; i += 2)
        {
            sc_start(1, SC_NS);
            if (!(dut.current_ram_index == i))
            {
                cout << "dut.current_ram_index == " << i << " FAILED!" << endl;
                return false;
            }
        }
        cout << "validate success!" << endl;

        // load cycle
        sc_start(1, SC_NS);

        // first cycle
        sc_start(1, SC_NS);

        cout << "validate second wait descriptor execution" << endl;
        for (unsigned int i = 3; i <= 9; i += 3)
        {
            sc_start(1, SC_NS);
            if (!(dut.current_ram_index == i))
            {
                cout << "dut.current_ram_index == " << i << " FAILED!" << endl;
                return false;
            }
        }
        cout << "validate success!" << endl;

        cout << "validate final descriptor is suspend" << endl;

        // load cycle
        sc_start(1, SC_NS);
        if (!(dut.currentDescriptor() == suspend_descriptor))
        {
            cout << "dut.currentDescriptor() == suspend_descriptor FAILED!" << endl;
            return false;
        }
        cout << "validate success!" << endl;

        cout << "validate_wait SUCCESS" << endl;
        return true;
    }
    bool validate_generation_1D()
    {
        cout << "Validating verify_generation_1D" << endl;
        control.set_enable(false);
        control.set_reset(true);
        sc_start(1, SC_NS);
        control.set_reset(false);
        sc_start(1, SC_NS);

        Descriptor_2D generate_1D_descriptor_1(1, 10, DescriptorState::GENERATE, 10, 2, 0, 0);
        Descriptor_2D generate_1D_descriptor_2(2, 20, DescriptorState::GENERATE, 10, -1, 0, 0);
        Descriptor_2D suspend_descriptor(1, 0, DescriptorState::SUSPENDED, 0, 0, 0, 0);
        vector<Descriptor_2D> temp_program;
        temp_program.push_back(generate_1D_descriptor_1);
        temp_program.push_back(generate_1D_descriptor_2);
        temp_program.push_back(suspend_descriptor);

        dut.loadProgram(temp_program);
        control.set_program(true);
        // load program and start first descriptor
        sc_start(1, SC_NS);
        control.set_program(false);
        control.set_enable(true);
        // first cycle (channel enable cycle)
        sc_start(1, SC_NS);
        if (!(mem_channel.addr() == 10))
        {
            cout << "mem_channel.addr() == 10 FAILED!" << endl;
            return false;
        }
        if (!(dut.currentDescriptor() == generate_1D_descriptor_1))
        {
            cout << "dut.currentDescriptor() == generate_1D_descriptor_1 FAILED!" << endl;
            return false;
        }

        cout << "validate success!" << endl;

        cout << "validate address 1D generation with positive xModify" << endl;
        for (unsigned int i = 12; i <= 30; i += 2)
        {
            sc_start(1, SC_NS);
            if (!(mem_channel.addr() == i))
            {
                cout << "mem_channel.addr() == " << i << " FAILED!" << endl;
                return false;
            }
        }
        cout << "validate success!" << endl;

        cout << "validate transition from first generate descriptor to second "
                "generate descriptor"
             << endl;

        // load cycle
        sc_start(1, SC_NS);

        // channel enable cycle
        sc_start(1, SC_NS);
        if (!(mem_channel.addr() == 20))
        {
            cout << "mem_channel.addr() == 20 FAILED!" << endl;
            return false;
        }
        if (!(dut.currentDescriptor() == generate_1D_descriptor_2))
        {
            cout << "dut.currentDescriptor() == generate_1D_descriptor_2 FAILED!" << endl;
            return false;
        }
        cout << "validate success!" << endl;

        cout << "validate address 1D generation with negative xModify" << endl;
        for (unsigned int i = 19; i >= 10; i -= 1)
        {
            sc_start(1, SC_NS);
            if (!(mem_channel.addr() == i))
            {
                cout << "mem_channel.addr() == " << i << " FAILED!" << endl;
                return false;
            }
            // if (!(mem_channel.addr() == i))
            // {
            //     cout << "mem_channel.addr() == i FAILED!" << endl;
            //     return false;
            // }
        }
        cout << "validate success!" << endl;

        cout << "verify_generation_1D SUCCESS" << endl;
        return true;
    }

    bool validate_generation_2D()
    {
        cout << "Validating verify_generation_2D" << endl;
        control.set_enable(false);
        control.set_reset(true);
        sc_start(1, SC_NS);
        control.set_reset(false);
        sc_start(1, SC_NS);

        Descriptor_2D generate_2D_descriptor_1(1, 10, DescriptorState::GENERATE, 10, 2, 5, 5);
        Descriptor_2D generate_2D_descriptor_2(2, 500, DescriptorState::GENERATE, 10, -1, 5, -5);
        Descriptor_2D suspend_descriptor(1, 0, DescriptorState::SUSPENDED, 0, 0, 0, 0);
        vector<Descriptor_2D> temp_program;
        temp_program.push_back(generate_2D_descriptor_1);
        temp_program.push_back(generate_2D_descriptor_2);
        temp_program.push_back(suspend_descriptor);

        dut.loadProgram(temp_program);
        control.set_program(true);
        // load program and start first descriptor
        sc_start(1, SC_NS);

        control.set_program(false);
        control.set_enable(true);

        // enable channel
        sc_start(1, SC_NS);

        bool first_cycle = true;
        // run descriptors
        cout << "validate address 2D generation with positive xModify and positive "
                "yModify"
             << endl;
        unsigned int i, j;

        int index = 10;
        for (i = 0; i <= 5; i++)
        {
            for (j = 0; j <= 10; j++)
            {
                if (first_cycle)
                {
                    cout << "\t validate initial condition at first cycle" << endl;
                    if (!(dut.currentDescriptor() == generate_2D_descriptor_1))
                    {
                        cout << "dut.currentDescriptor() == generate_2D_descriptor_1 FAILED!" << endl;
                        return false;
                    }
                    cout << "\t validate success!" << endl;
                    first_cycle = false;
                }
                if (!(mem_channel.addr() == (unsigned int)index))
                {
                    cout << "mem_channel.addr() == index FAILED!" << endl;
                    return false;
                }
                if (j != 10)
                {
                    index += 2;
                }
                sc_start(1, SC_NS);
            }
            index += 5;
        }

        cout << "validate success!" << endl;

        // enable channel
        sc_start(1, SC_NS);

        cout << "validate address 2D generation with negative xModify and negative "
                "yModify"
             << endl;
        first_cycle = true;
        index = 500;
        for (i = 0; i <= 5; i++)
        {
            for (j = 0; j <= 10; j++)
            {
                if (first_cycle)
                {
                    cout << "\t validate transition from first generate descriptor to second "
                            "generate descriptor"
                         << endl;
                    if (!(dut.currentDescriptor() == generate_2D_descriptor_2))
                    {
                        cout << "dut.currentDescriptor() == generate_2D_descriptor_2 FAILED!" << endl;
                        return false;
                    }
                    cout << "\t validate success!" << endl;

                    first_cycle = false;
                }
                if (!(mem_channel.addr() == (unsigned int)index))
                {
                    cout << "mem_channel.addr() == index FAILED!" << endl;
                    return false;
                }
                if (j != 10)
                {
                    index -= 1;
                }
                sc_start(1, SC_NS);
            }
            index -= 5;
        }
        cout << "validate success!" << endl;

        cout << "verify_generation_2D SUCCESS" << endl;
        return true;
    }

    int run_tb()
    {

        if (!validate_reset())
        {
            cout << "validate_reset() FAILED!" << endl;
            return -1;
        }

        if (!(validate_loadprogram_and_suspended_state()))
        {
            cout << "validate_loadprogram_and_suspended_state() FAILED!" << endl;
            return -1;
        }

        if (!(validate_wait_and_descriptor_retirement()))
        {
            cout << "validate_wait() FAILED!" << endl;
            return -1;
        }

        if (!(validate_generation_1D()))
        {
            cout << "validate_generation_1D() FAILED!" << endl;
            return -1;
        }

        if (!(validate_generation_2D()))
        {
            cout << "validate_generation_2D() FAILED!" << endl;
            return -1;
        }

        cout << "ALL TESTS PASS" << endl;

        return 0;
    }
    ~AddressGenerator_TB()
    {
        sc_close_vcd_trace_file(tf);
    }
};
int sc_main(int argc, char *argv[])
{
    AddressGenerator_TB<sc_int<32>> tb("AddressGenerator_tb");
    return tb.run_tb();
}
