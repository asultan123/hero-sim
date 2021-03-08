#ifndef ComputeBlob_CPP
#define ComputeBlob_CPP

#include <GlobalControl.cpp>
#include <assert.h>
#include <iostream>
#include <string>
#include <systemc.h>

using std::cout;
using std::endl;
using std::string;
template <typename DataType>
struct ComputeBlob : public sc_module
{
    // Member Signals
private:
    sc_in_clk _clk;

public:
    sc_port<GlobalControlChannel_IF> control;
    sc_vector<sc_signal<DataType>> weight{"weights", 27};
    sc_vector<sc_signal<DataType>> psums{"psums", 28};
    sc_vector<sc_in<DataType>> pe_group_in{"pe_group_in", 9};
    sc_out<DataType> psum_out{"psum_out"};
    sc_trace_file* tf;

    void update()
    {
        if (control->reset())
        {
            for (int i = 0; i < 27; i++)
            {
                weight[i] = 0;
                psums[i] = 0;
            }
            psums[27] = 0;
        }
        else if (control->enable())
        {
            for (int pe_group = 0; pe_group < 9; pe_group++)
            {
                // cout << "pe_group[" << pe_group << "]: " << pe_group_in[pe_group].read() << endl;
                for (int pe = 0; pe < 3; pe++)
                {
                    psums[pe_group * 3 + pe + 1] = psums[pe_group * 3 + pe].read() + (weight[pe_group * 3 + pe].read() * pe_group_in[pe_group].read());
                }
            }
        }
    }

    void loadWeights(const std::vector<DataType>& values)
    {
        for (int i = 0; i < 27; i++)
        {
            weight[i] = values[i];
        }
    }

    // Constructor
    ComputeBlob(
        sc_module_name name,
        GlobalControlChannel& _control,
        sc_vector<sc_signal<DataType>>& _pe_group_in,
        sc_trace_file* _tf) : sc_module(name)
    {
        control(_control);
        _clk(control->clk());
        psum_out.bind(psums[27]);
        tf = _tf;
        for (int i = 0; i < 9; i++)
        {
            pe_group_in[i].bind(_pe_group_in[i]);
        }

        sc_trace(tf, pe_group_in[0], "pe_group_in_0");
        sc_trace(tf, pe_group_in[1], "pe_group_in_1");
        sc_trace(tf, pe_group_in[2], "pe_group_in_2");
        sc_trace(tf, pe_group_in[3], "pe_group_in_3");
        sc_trace(tf, pe_group_in[4], "pe_group_in_4");
        sc_trace(tf, pe_group_in[5], "pe_group_in_5");
        sc_trace(tf, pe_group_in[6], "pe_group_in_6");
        sc_trace(tf, pe_group_in[7], "pe_group_in_7");
        sc_trace(tf, pe_group_in[8], "pe_group_in_8");

        for (int i = 0; i < 28; i++)
        {
            sc_trace(tf, psums[i], (string("psums") + std::to_string(i)));
        }
        SC_METHOD(update);
        sensitive << _clk.pos();
        sensitive << control->reset();
        cout << "ComputeBlob MODULE: " << name << " has been instantiated " << endl;
    }

    SC_HAS_PROCESS(ComputeBlob);
};

template <typename DataType>
struct ComputeBlobInjector : public sc_module
{
    // Member Signals
private:
    sc_in_clk _clk;
    sc_uint<32> timeval{0};
    sc_port<GlobalControlChannel_IF> control;

public:
    sc_vector<sc_signal<DataType>> pe_group_sig;
    ComputeBlob<DataType> blob;

    void update()
    {
        std::vector<DataType> ifmap(300);
        for (int val = 0; val < 300; val++)
        {
            ifmap[val] = val + 1;
        }
        while (1)
        {
            if (control->enable())
            {
                std::vector<std::vector<int>> slice(3, std::vector<int>(3, -1));
                for(int filter = 0; filter < 3; filter++)
                {
                    for(int ifmapIdx = 0; ifmapIdx <= /*numbers sent to each input*/80+/*max of startup delay for all inputs*/3*8; ifmapIdx++)
                    {
                        cout << "ifmapIdx: " << ifmapIdx << endl;
                        for (int chIdx = 0; chIdx < 3; chIdx++)
                        {
                            for (int grpIdx = 0; grpIdx < 3; grpIdx++)
                            {
                                int groupStartTime = (chIdx * 9 + grpIdx*3);
                                int bytesToRead = 80;
                                if (ifmapIdx >= groupStartTime && ifmapIdx < (bytesToRead + groupStartTime))
                                {
                                    int grp_ifmap_idx = (ifmapIdx - (chIdx * 9 + grpIdx*3)) + grpIdx * 10 + chIdx * 100;
                                    pe_group_sig[chIdx * 3 + grpIdx].write(ifmap[grp_ifmap_idx]);
                                    slice[chIdx][grpIdx] = grp_ifmap_idx;
                                }
                            }
                        }
                        for (auto& row : slice)
                        {
                            for (auto& col : row)
                            {
                                cout << col << ", ";
                            }
                            cout << endl;
                        }
                        wait();
                    }
                }
                // timeval++;
            }
            wait();
        }
    }

    void loadBlobWeights()
    {
        std::vector<DataType> weights(27);
        for (int j = 0; j < 3; j++)
        {
            for (int i = 0; i < 9; i++)
            {
                weights[j * 9 + i] = i + 1;
            }
        }
        blob.loadWeights(weights);
    }

    // Constructor
    ComputeBlobInjector(
        sc_module_name name,
        GlobalControlChannel& _control,
        sc_trace_file* tf) : sc_module(name),
                             pe_group_sig("pe_group_sig", 9),
                             blob("blob", _control, pe_group_sig, tf)
    {
        control(_control);
        _clk(control->clk());
        SC_THREAD(update);
        sensitive << _clk.pos();
        sensitive << control->reset();
        cout << "ComputeBlobInjector MODULE: " << name << " has been instantiated " << endl;
    }

    SC_HAS_PROCESS(ComputeBlobInjector);
};

template <typename DataType>
struct ComputeBlob_TB : public sc_module
{
    sc_trace_file* tf{sc_create_vcd_trace_file("computeblob")};
    GlobalControlChannel control{"global_control_channel", sc_time(1, SC_NS), tf};
    ComputeBlobInjector<DataType> dut{"Injector", control, tf};

    ComputeBlob_TB(sc_module_name name) : sc_module(name)
    {
        tf->set_time_unit(1, SC_PS);
        cout << "Instantiated ComputeBlob TB with name " << this->name() << endl;
    }
    bool validate_reset()
    {
        control.set_reset(true);
        control.set_enable(false);
        sc_start(5, SC_NS);

        for (int i = 0; i < 27; i++)
        {
            if (dut.blob.weight[i].read() != DataType(0))
            {
                cout << "dut.blob.weights[i] != DataType(0) FAILED!" << endl;
                return false;
            }
        }

        control.set_reset(false);
        dut.loadBlobWeights();
        sc_start(1, SC_NS);
        for (int i = 0; i < 3; i++)
        {
            for (int j = 0; j < 9; j++)
            {
                cout << "expected: " << DataType(j + 1) << " actual: " << dut.blob.weight[i * 9 + j] << endl;
                if (dut.blob.weight[i * 9 + j] != DataType(j + 1))
                {
                    cout << "dut.blob.weight[i] != i+1 FAILED!" << endl;
                    return false;
                }
            }
        }
        return true;
    }

    bool validate_injection()
    {
        cout << "Validating injection" << endl;
        control.set_enable(true);
        float expected_output[64] = {15678, 15813, 15948, 16083, 16218, 16353, 16488, 16623, 17028, 17163, 17298, 17433, 17568, 17703, 17838, 17973, 18378, 18513, 18648, 18783, 18918, 19053, 19188, 19323, 19728, 19863, 19998, 20133, 20268, 20403, 20538, 20673, 21078, 21213, 21348, 21483, 21618, 21753, 21888, 22023, 22428, 22563, 22698, 22833, 22968, 23103, 23238, 23373, 23778, 23913, 24048, 24183, 24318, 24453, 24588, 24723, 25128, 25263, 25398, 25533, 25668, 25803, 25938, 26073};
        int success_count = 0;
        for(int filter = 0; filter < 3; filter++)
        {
            bool startValidation = false;
            int expected_output_index = 0;
            int validCounter = 8;
            int invalidCounter = 1;
            for (int k = 0; expected_output_index < 64; ++k)
            {
                sc_start(1, SC_NS);
                cout << "psumout " << dut.blob.psum_out.read() << endl;
                if (!startValidation && DataType(expected_output[0]) == dut.blob.psum_out.read())
                {
                    startValidation = true;
                }
                if (startValidation)
                {
                    cout << "Current expected output index: " << expected_output_index << endl;
                    if (validCounter > 0)
                    {
                        std::cout << "@" << sc_time_stamp() << " dut.blob.psum_out: " << dut.blob.psum_out
                                << " expected_output: " << expected_output[expected_output_index];

                        if (DataType(expected_output[expected_output_index]) == dut.blob.psum_out.read())
                        {
                            std::cout << " ....assertion succuss!!" << std::endl;
                            expected_output_index++;
                        }
                        else
                        {
                            std::cout << " ....assertion failure, terminating....!!" << std::endl;
                            break;
                        }
                        validCounter--;
                    }
                    else if (invalidCounter > 0)
                    {
                        std::cout << "@" << sc_time_stamp() << " dut.blob.psum_out: " << dut.blob.psum_out
                                << " ....ignoring invalid output " << std::endl;
                        invalidCounter--;
                    }
                    else
                    {
                        std::cout << "@" << sc_time_stamp() << " dut.blob.psum_out: " << dut.blob.psum_out
                                << " ....ignoring invalid output " << std::endl;
                        validCounter = 8;
                        invalidCounter = 1;
                    }
                }
            }

            if (expected_output_index == 64)
            {
                cout << "filter " << filter << " SUCCESS" << endl;
                success_count++;
            }
            else
            {
                cout << "filter " << filter << " SUCCESS" << endl;
                return false;
            }
        }

        if(success_count == 3)
        {
            return true;
        }
        else
        {
            return false;
        }
        
    }

    int run_tb()
    {
        cout << "Validating Reset" << endl;
        if (!validate_reset())
        {
            cout << "Reset Failed" << endl;
            return -1;
        }

        if (!validate_injection())
        {
            cout << "!validate_injection() FAILED!" << endl;
            return -1;
        }

        cout << "TEST BENCH SUCCESS " << endl;
        cout << "       aOOOOOOOOOOa" << endl;
        cout << "     aOOOOOOOOOOOOOOa" << endl;
        cout << "   aOO    OOOOOO    OOa" << endl;
        cout << "  aOOOOOOOOOOOOOOOOOOOa" << endl;
        cout << " aOOOOO   OOOOOO   OOOOOa" << endl;
        cout << "aOOOOO     OOOO     OOOOOa" << endl;
        cout << "aOOOOOOOOOOOOOOOOOOOOOOOOa" << endl;
        cout << "aOOOOOOOOOOOOOOOOOOOOOOOOa" << endl;
        cout << "aOOOOO   OOOOOOOO   OOOOOa" << endl;
        cout << " aOOOOO    OOOO    OOOOOa" << endl;
        cout << "  aOOOOO          OOOOOa" << endl;
        cout << "   aOOOOOOOOOOOOOOOOOOa" << endl;
        cout << "     aOOOOOOOOOOOOOOa" << endl;
        cout << "       aOOOOOOOOOOa" << endl;
        return 0;
    }
    ~ComputeBlob_TB()
    {
        sc_close_vcd_trace_file(tf);
    }
};
int sc_main(int argc, char* argv[])
{
    ComputeBlob_TB<sc_int<32>> tb("ComputeBlob_tb");
    return tb.run_tb();
}

#endif // MEM_HIERARCHY_CPP
