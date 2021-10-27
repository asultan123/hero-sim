#ifndef __ESTIMATION_ENVIORNMENT_CC
#define __ESTIMATION_ENVIORNMENT_CC

#include "GlobalControl.hh"
#include <assert.h>
#include <iostream>
#include <string>
#include <systemc.h>
#include <sstream>
#include "pe.hh"

using std::cout;
using std::endl;
using std::string;

template <typename DataType>
struct PeCreator
{
    PeCreator(sc_trace_file* _tf)
    {
        tf = _tf;
    }
    PE<DataType>* operator()(const char* name, size_t)
    {
        return new PE<DataType>(name, this->tf);
    }
    sc_trace_file* tf;
};

template <typename DataType>
struct Arch_1x1 : public sc_module
{
    // Member Signals
private:
    sc_in_clk _clk;

public:
    sc_port<GlobalControlChannel_IF> control;
    // sc_vector<PE<DataType>> pe_array{"pe_array", 63, PeCreator<DataType>(tf)};
    sc_vector<sc_signal<DataType>> filter_psum_out{"psum_out", 7};
    sc_trace_file *tf;

    void update()
    {
        if (control->enable())
        {
            // for (int filter_row = 0; filter_row < 7; filter_row++)
            // {
            //     for (int channel_column = 0; channel_column < 8; channel_column++)
            //     {
            //         PE<DataType> *cur_pe = this->pe_array[filter_row * 9 + channel_column];
            //         PE<DataType> *next_pe = this->pe_array[filter_row * 9 + channel_column + 1];
            //         next_pe.psum_in = cur_pe->psum_in.read() + cur_pe->current_weight() * filter_row * 9 + channel_column;
            //         cur_pe.updateState();
            //     }
            //     PE<DataType> *last_pe = this->pe_array[filter_row * 9 + 8];
            //     filter_psum_out[filter_row] = last_pe->psum_in.read() + last_pe->current_weight() * filter_row * 9 + 8;
            //     last_pe->updateState();
            // }
        }
    }

    void load_weights(vector<vector<vector<int>>> weights)
    {
        for (int filter_row = 0; filter_row < 7; filter_row++)
        {
            for (int channel_column = 0; channel_column < 8; channel_column++)
            {
                PE<DataType> *cur_pe = this->pe_array[filter_row * 9 + channel_column];
                cur_pe->loadWeights(weights[filter_row][channel_column]);
            }
        }
    }

    void load_pe_program(vector<vector<vector<Descriptor_2D>>> programs)
    {
        for (int filter_row = 0; filter_row < 7; filter_row++)
        {
            for (int channel_column = 0; channel_column < 8; channel_column++)
            {
                PE<DataType> *cur_pe = this->pe_array[filter_row * 9 + channel_column];
                cur_pe->loadProgram(programs[filter_row][channel_column]);
            }
        }
    }

    // Constructor
    Arch_1x1(
        sc_module_name name,
        GlobalControlChannel &_control,
        sc_trace_file *_tf) : sc_module(name), tf(_tf)
    {
        control(_control);
        _clk(control->clk());
        tf = _tf;


        SC_METHOD(update);
        sensitive << _clk.pos();
        sensitive << control->reset();
        cout << "Arch_1x1 MODULE: " << name << " has been instantiated " << endl;
    }

    SC_HAS_PROCESS(Arch_1x1);
};


int sc_main(int argc, char *argv[])
{
    sc_trace_file* tf = sc_create_vcd_trace_file("ProgTrace");
    tf->set_time_unit(1, SC_PS);
    GlobalControlChannel control("global_control_channel", sc_time(1, SC_NS), tf);
    Arch_1x1<int> arch("arch", control, tf);
    PE<float> ape("ape", tf);

    cout << "HELLO WORLD!" << endl;
    // ComputeBlob_TB<sc_int<32>> tb("ComputeBlob_tb");
    return 0;
    // return tb.run_tb();
}

#endif // MEM_HIERARCHY_CPP
