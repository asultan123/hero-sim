#include "AddressGenerator.hh"
#include "SAM.hh"
#include <map>
#include <systemc>
#include <memory>

// #define DEBUG
using std::cout;
using std::endl;
using std::map;
using std::unique_ptr;

template <typename DataType>
struct SAM_TB : public sc_module
{
    const unsigned int dut_mem_length = 128;
    const unsigned int dut_mem_width = 4;
    const unsigned int dut_mem_channel_count = 2;
    sc_trace_file* tf;
    GlobalControlChannel control;
    SAM<DataType> dut;

    sc_vector<sc_signal<DataType>> external_channel_0_read_bus;
    sc_vector<sc_signal<DataType>> external_channel_0_write_bus;
    sc_vector<sc_signal<DataType>> external_channel_1_read_bus;
    sc_vector<sc_signal<DataType>> external_channel_1_write_bus;

    SAM_TB(sc_module_name name) : sc_module(name),
                                  tf(sc_create_vcd_trace_file("ProgTrace")),
                                  control("global_control_channel", sc_time(1, SC_NS), tf),
                                  dut("dut", control, dut_mem_channel_count, dut_mem_length, dut_mem_width, tf),
                                  external_channel_0_read_bus("external_channel_0_read_bus", dut_mem_width),
                                  external_channel_0_write_bus("external_channel_0_write_bus", dut_mem_width),
                                  external_channel_1_read_bus("external_channel_1_read_bus", dut_mem_width),
                                  external_channel_1_write_bus("external_channel_1_write_bus", dut_mem_width)
                                  
    {
        for (unsigned int i = 0; i < dut_mem_width; i++)
        {
            dut.read_channel_data[0][i](external_channel_0_read_bus[i]);
            dut.write_channel_data[0][i](external_channel_0_write_bus[i]);
            dut.read_channel_data[1][i](external_channel_1_read_bus[i]);
            dut.write_channel_data[1][i](external_channel_1_write_bus[i]);

            sc_trace(tf, external_channel_0_read_bus[i], external_channel_0_read_bus[i].name());
            sc_trace(tf, external_channel_0_write_bus[i], external_channel_0_write_bus[i].name());
            sc_trace(tf, external_channel_1_read_bus[i], external_channel_1_read_bus[i].name());
            sc_trace(tf, external_channel_1_write_bus[i], external_channel_1_write_bus[i].name());
        }
        tf->set_time_unit(1, SC_PS);
        cout << "Instantiated SAM TB with name " << this->name() << endl;
    }
    bool validate_reset()
    {
        control.set_reset(true);

        control.set_program(false);
        control.set_enable(false);

        sc_start(1, SC_NS);

        control.set_reset(false);

        sc_start(1, SC_NS);

        for (unsigned int idx = 0; idx < dut_mem_channel_count; idx++)
        {
            cout << "checking address generator[" << idx << "] " << endl;
            if (!(dut.generators[idx].descriptors[0] == Descriptor_2D::default_descriptor()))
            {
                cout << "dut.generators[idx].descriptors[0] == default_descriptor FAILED!" << endl;
                return false;
            }

            if (!(dut.generators[idx].execute_index == 0))
            {
                cout << "dut.generators[idx].execute_index == 0 FAILED!" << endl;
                return false;
            }

            if (!(dut.generators[idx].programmed == false))
            {
                cout << "dut.generators[idx].programmed == false FAILED!" << endl;
                return false;
            }
            cout << "address generator[" << idx << "] reset correctly! " << endl;

            cout << "checking channel[" << idx << "] " << endl;
            if (!(dut.generators[idx].descriptors[0] == Descriptor_2D::default_descriptor()))
                for (auto& data : dut.channels[idx].read_channel_data)
                {
                    if (!(data == DataType(0)))
                    {
                        cout << "channel read_bus_data == 0 FAILED!" << endl;
                        return false;
                    }
                }
            for (auto& data : dut.channels[idx].write_channel_data)
            {
                if (!(data == DataType(0)))
                {
                    cout << "channel write_bus_data == 0 FAILED!" << endl;
                    return false;
                }
            }
            cout << "channel[" << idx << "] reset correctly! " << endl;
        }

        // check memory cells
        for (auto& row : dut.mem.ram)
        {
            for (auto& col : row)
            {
                if (!(col == DataType(0)))
                {
                    cout << "col == DataType(0) FAILED!" << endl;
                    return false;
                }
            }
        }

        sc_start(20, SC_NS);

        return true;
    }

    bool validate_write_to_sam_1D()
    {
        cout << "Validating validate_write_to_sam_1D" << endl;

        control.set_reset(true);
        control.set_program(false);
        control.set_enable(false);

        sc_start(1, SC_NS);

        control.set_reset(false);

        Descriptor_2D generate_1D_descriptor_1(1, 10, DescriptorState::GENERATE, 9,
                                               1, 0, 0);

        Descriptor_2D suspend_descriptor(1, 0, DescriptorState::SUSPENDED, 0, 0, 0,
                                         0);

        vector<Descriptor_2D> temp_program;
        temp_program.push_back(generate_1D_descriptor_1);
        temp_program.push_back(suspend_descriptor);

        dut.generators[0].loadProgram(temp_program);
        dut.channels[0].set_mode(MemoryChannelMode::WRITE);
        dut.channels[1].set_mode(MemoryChannelMode::READ);

        control.set_program(true);
        cout << "load program and start first descriptor" << endl;
        sc_start(1, SC_NS);
        control.set_enable(true);
        control.set_program(false);

        external_channel_0_write_bus[0] = DataType(1);
        sc_start(1, SC_NS); 

        for (unsigned int i = 2; i <= 10; i++)
        {
            external_channel_0_write_bus[0] = DataType(i);
            sc_start(1, SC_NS);
        }
        // write to ram always happens 1 cycles after external data is set so one last
        // clk pulse is needed to validate change 
        sc_start(1, SC_NS); 

        int expected_val = 1;
        for(int i = 10; i<20; i++)
        {
            if(dut.mem.ram[i][0] != DataType(expected_val))
            {
                cout << "dut.mem.ram[i][0] != expected_val: " << expected_val << "FAILED!" << endl;
                return false;
            }
            expected_val++;
        }

        cout << "validate_write_to_sam_1D SUCCESS" << endl;

        sc_start(20, SC_NS);
    
        return true;
    }

    bool validate_write_to_sam_2D()
    {
        cout << "Validating validate_write_to_sam_2D" << endl;

        cout << "validate_write_to_sam_2D SUCCESS" << endl;
        return true;
    }

    bool validate_read_from_sam_1D()
    {
        cout << "Validating validate_read_from_sam_1D" << endl;

        control.set_reset(true);
        control.set_program(false);
        control.set_enable(false);

        sc_start(1, SC_NS);

        control.set_reset(false);

        Descriptor_2D generate_1D_descriptor_1(1, 10, DescriptorState::GENERATE, 10,
                                               1, 0, 0);

        Descriptor_2D suspend_descriptor(1, 0, DescriptorState::SUSPENDED, 0, 0, 0,
                                         0);

        vector<Descriptor_2D> temp_program;
        temp_program.push_back(generate_1D_descriptor_1);
        temp_program.push_back(suspend_descriptor);

        dut.generators[1].loadProgram(temp_program);
        dut.channels[0].set_mode(MemoryChannelMode::WRITE);
        dut.channels[1].set_mode(MemoryChannelMode::READ);

        unsigned int index = 1;
        for (unsigned int row = 0; row < dut_mem_length; row++)
        {
            for (unsigned int col = 0; col < dut_mem_width; col++)
            {
                dut.mem.ram[row][col] = index++;
            }
        }
        sc_start(1, SC_NS);

        control.set_program(true);
        cout << "load program and start first descriptor" << endl;
        sc_start(1, SC_NS);
        control.set_enable(true);
        control.set_program(false);

        // after the cycle so that wire can update. update happens after an
        // infinitesimally small amount of time beyond rising edge but for
        // simplicity I'm jumping half a cycle so that other testcases can
        // adjust
        sc_start(1.5, SC_NS); 

        // generate descriptor reads 10 rows of size 4 and outputs them to read.
        // Bus width always matches memory width (otherwise you'd something to marshal)
        for (unsigned int i = 40; i <= 80; i+=4)
        {
            if(external_channel_1_read_bus[0] != DataType(i+1))
            {
                cout << "external_channel_1_read_bus[0] != " << i+1 << " FAILED!" << endl;
                return false;
            }
            if(external_channel_1_read_bus[1] != DataType(i+2))
            {
                cout << "external_channel_1_read_bus[1] != " << i+2 << " FAILED!" << endl;
                return false;
            }
            if(external_channel_1_read_bus[2] != DataType(i+3))
            {
                cout << "external_channel_1_read_bus[2] != " << i+3 << " FAILED!" << endl;
                return false;
            }
            if(external_channel_1_read_bus[3] != DataType(i+4))
            {
                cout << "external_channel_1_read_bus[3] != " << i+4 << " FAILED!" << endl;
                return false;
            }                        
            sc_start(1, SC_NS);
        }

        sc_start(20, SC_NS);

        cout << "validate_read_from_sam_1D SUCCESS" << endl;
        return true;
    }

    bool validate_read_from_sam_2D()
    {
        cout << "Validating validate_read_from_sam_2D" << endl;

        cout << "validate_read_from_sam_2D SUCCESS" << endl;
        return true;
    }

    bool validate_wait_with_data_write()
    {
        cout << "Validating validate_wait_with_data_write" << endl;

        control.set_reset(true);
        control.set_program(false);
        control.set_enable(false);

        sc_start(1, SC_NS);

        control.set_reset(false);

        Descriptor_2D wait_descriptor(1, 10, DescriptorState::WAIT, 10,
                                      1, 0, 0);

        Descriptor_2D suspend_descriptor(1, 0, DescriptorState::SUSPENDED, 0, 0, 0,
                                         0);

        vector<Descriptor_2D> temp_program;
        temp_program.push_back(wait_descriptor);
        temp_program.push_back(suspend_descriptor);

        dut.generators[0].loadProgram(temp_program);
        dut.channels[0].set_mode(MemoryChannelMode::WRITE);
        dut.channels[1].set_mode(MemoryChannelMode::READ);

        control.set_program(true);
        cout << "load program and start first descriptor" << endl;
        sc_start(1, SC_NS);
        control.set_enable(true);
        control.set_program(false);

        for (unsigned int i = 1; i <= 10; i++)
        {
            // dut.write_channel_data[0][0]->write(i);
            external_channel_0_write_bus[0] = DataType(i);
            if (!(dut.channels[0].enabled() == false))
            {
                cout << "dut.channels[0].enabled() == false FAILED!" << endl;
                return false;
            }
            sc_start(1, SC_NS);
        }
        sc_start(10, SC_NS);

        for (unsigned int row = 0; row < dut_mem_length; row++)
        {
            for (unsigned int col = 0; col < dut_mem_width; col++)
            {
                if (!(dut.mem.ram[row][col] == DataType(0)))
                {
                    cout << "dut.mem.ram[row][col] == 0 FAILED!" << endl;
                    return false;
                }
            }
        }
        cout << "validate_wait_with_data_write SUCCESS" << endl;
        sc_start(20, SC_NS);

        return true;
    }

    bool validate_write_then_read_after_wait_1D()
    {
        cout << "Validating validate_write_then_read_after_wait_1D" << endl;

        control.set_reset(true);
        control.set_program(false);
        control.set_enable(false);

        sc_start(1, SC_NS);

        control.set_reset(false);

        Descriptor_2D generator_0_write_1D_descriptor(1, 10, DescriptorState::GENERATE, 9,
                                                      1, 0, 0);

        Descriptor_2D generator_0_write_suspend_descriptor(1, 0, DescriptorState::SUSPENDED, 0, 0, 0,
                                                           0);

        vector<Descriptor_2D> temp_program;
        temp_program.push_back(generator_0_write_1D_descriptor);
        temp_program.push_back(generator_0_write_suspend_descriptor);

        dut.generators[0].loadProgram(temp_program);

        temp_program.clear();

        // execution adds 1 cycle by default, loading consumes 1 cycle as well
        Descriptor_2D generator_1_wait_descriptor(1, 10, DescriptorState::WAIT, 2,
                                                  5, 0, 0);

        Descriptor_2D generator_1_read_1D_descriptor(2, 10, DescriptorState::GENERATE, 9,
                                                     1, 0, 0);

        Descriptor_2D generator_1_read_suspend_descriptor(2, 0, DescriptorState::SUSPENDED, 0, 0, 0,
                                                          0);

        temp_program.push_back(generator_1_wait_descriptor);
        temp_program.push_back(generator_1_read_1D_descriptor);
        temp_program.push_back(generator_1_read_suspend_descriptor);

        dut.generators[1].loadProgram(temp_program);

        dut.channels[0].set_mode(MemoryChannelMode::WRITE);
        dut.channels[1].set_mode(MemoryChannelMode::READ);

        control.set_program(true);
        cout << "load program and start first descriptor" << endl;

        sc_start(1, SC_NS);
        control.set_enable(true);
        control.set_program(false);
        sc_start(0.5, SC_NS); // time offset

        int expected_value = 1;
        for (unsigned int i = 1; i <= 15; i++)
        {
            if(i <= 10)
            {
                // stimulus
                external_channel_0_write_bus[0] = DataType(i);
                sc_start(1, SC_NS);

                // monitor
                // ignore 5th iteration because channeled is enabled there
                if(i < 5) 
                {
                    if(dut.channels[1].enabled() != false)
                    {
                        cout << "read channel enabled during wait! FAILED!" << endl;
                        return false;
                    }
                }
                // bus updated "after" 5th iteration (falling edge of current
                // cycle, see vcd output)
                else if(i > 5)
                {
                    if(external_channel_1_read_bus[0] != DataType(expected_value))
                    {
                        cout << "@" << sc_time_stamp() << " external_channel_1_read_bus != " << expected_value << " FAILED!" << endl;
                        cout << "@" << sc_time_stamp() << " " << i << endl;
                        return false;
                    }
                    expected_value++;
                }
            }
            else
            {
                sc_start(1, SC_NS);
                // no stimulus after 10 cycles because write address generator is now idle
                if(dut.channels[0].enabled() != false)
                {
                    cout << "Write generator failed to disable write channel upon being suspended" << endl;
                    return false;
                }
                if(external_channel_1_read_bus[0] != DataType(expected_value))
                {
                    cout << "read bus expected value validation FAILED!" << endl;
                    return false;
                }
                expected_value++;
            }
        }



        sc_start(20, SC_NS);

        cout << "validate_write_then_read_after_wait_1D SUCCESS" << endl;
        return true;
    }

    bool validate_concurrent_read_write_1D()
    {
        cout << "Validating validate_concurrent_read_write_1D" << endl;

        control.set_reset(true);
        control.set_program(false);
        control.set_enable(false);

        // clear bus for gtk
        external_channel_1_read_bus[0].write(0);

        sc_start(1.5, SC_NS);

        control.set_reset(false);

        Descriptor_2D generator_0_write_1D_descriptor_1(1, 10, DescriptorState::GENERATE, 9,
                                                        1, 0, 0);

        Descriptor_2D generator_0_write_suspend_descriptor(1, 0, DescriptorState::SUSPENDED, 0, 0, 0,
                                                           0);

        vector<Descriptor_2D> temp_program;
        temp_program.push_back(generator_0_write_1D_descriptor_1);
        temp_program.push_back(generator_0_write_suspend_descriptor);

        dut.generators[0].loadProgram(temp_program);

        temp_program.clear();

        // need wait, otherwise will read before write
        // wait execution minimum is 1 cycle
        // loading next descriptor is another cycle
        // decoding next descriptor (enabling channel in case of generate) is another cycle
        Descriptor_2D generator_1_wait_descriptor(1, 20, DescriptorState::WAIT, 0,
                                                  0, 0, 0);

        Descriptor_2D generator_1_read_1D_descriptor_1(2, 10, DescriptorState::GENERATE, 9,
                                                       1, 0, 0);

        Descriptor_2D generator_1_read_suspend_descriptor(2, 0, DescriptorState::SUSPENDED, 0, 0, 0,
                                                          0);

        temp_program.push_back(generator_1_wait_descriptor);
        temp_program.push_back(generator_1_read_1D_descriptor_1);
        temp_program.push_back(generator_1_read_suspend_descriptor);

        dut.generators[1].loadProgram(temp_program);

        dut.channels[0].set_mode(MemoryChannelMode::WRITE);
        dut.channels[1].set_mode(MemoryChannelMode::READ);

        control.set_program(true);
        cout << "load program and start first descriptor" << endl;

        sc_start(1, SC_NS);
        control.set_enable(true);
        control.set_program(false);

        int expected_val = 1;
        for (unsigned int i = 1; i <= 12; i++)
        {
            external_channel_0_write_bus[0] = DataType(i);
            sc_start((i==1)? 1.5: 1, SC_NS); //offset bump for clarity

            // For read channel, 1 cycle for wait + 1 cycle for loading generate
            // @ 3rd cycle (i == 2) generator must begin executing generate descriptor
            // and enabled channel
            if(i < 2)
            {
                if(dut.channels[1].enabled() != false)
                {
                    cout << "read channel must be disabled during read generator wait" << endl;
                    cout << i << endl;
                    return false;
                }                
            }
            // at cycle 3 (i == 2) read channel has just been enabled, read values will only appear on bus @ cycle 4 (i == 3)
            else if(i>=3) 
            {
                if(external_channel_1_read_bus[0] != DataType(expected_val))
                {
                    cout << "read bus readback wrong value, expected: " << expected_val << endl;
                    return false;
                }
                if(i <= 11)
                {
                    if(dut.channels[1].enabled() != true)
                    {
                        cout << "read channel must enabled during generator address generation" << endl;
                        cout << i << endl;
                        return false;
                    }
                }
                // @ i = 12, read generator should have finished the generator descriptor and disabled channel
                else
                {
                    if(dut.channels[1].enabled() != false)
                    {
                        cout << "read channel must not be enabled during read generator suspend" << endl;
                        return false;
                    }
                }
                expected_val++;
            }
        }

        sc_start(20, SC_NS);

        cout << "validate_concurrent_read_write_1D SUCCESS" << endl;
        return true;
    }

    bool validate_concurrent_read_write_2D()
    {
        cout << "Validating validate_concurrent_read_write_2D" << endl;

        cout << "validate_concurrent_read_write_2D SUCCESS" << endl;
        return true;
    }

    int run_tb()
    {
        cout << "Validating Reset" << endl;
        if (!validate_reset())
        {
            cout << "Reset Failed" << endl;
            return false;
        }
        if (!(validate_write_to_sam_1D()))
        {
            cout << "validate_write_to_sam_1D() FAILED!" << endl;
            return false;
        }
        if (!(validate_read_from_sam_1D()))
        {
            cout << "validate_read_from_sam_1D() FAILED!" << endl;
            return false;
        }
        if(!(validate_wait_with_data_write()))
        {
            cout << "validate_wait_with_data_write() FAILED!" << endl;
            return false;
        }
        if (!(validate_concurrent_read_write_1D()))
        {
            cout << "validate_concurrent_read_write_1D() FAILED!" << endl;
            return false;
        }
        if (!(validate_write_then_read_after_wait_1D()))
        {
            cout << "validate_write_then_read_after_wait_1D() FAILED!" << endl;
            return false;
        }


        cout << "Reset Success" << endl;
        cout << "ALL TESTS PASS" << endl;
        return 0;
    }
    ~SAM_TB()
    {
        sc_close_vcd_trace_file(tf);
    }
};

int sc_main(int argc, char* argv[])
{
    SAM_TB<sc_int<32>> tb("SAM_tb");

    return tb.run_tb();
}
