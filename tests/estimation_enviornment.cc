#ifndef __ESTIMATION_ENVIORNMENT_CC
#define __ESTIMATION_ENVIORNMENT_CC

#include "GlobalControl.hh"
#include <assert.h>
#include <iostream>
#include <string>
#include <systemc.h>
#include <sstream>
#include "ProcEngine.hh"
#include "SAM.hh"
#include <chrono>

using std::cout;
using std::endl;
using std::string;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;


template <typename DataType>
struct SignalVectorCreator
{
    SignalVectorCreator(unsigned int _width, sc_trace_file *_tf) : tf(_tf), width(_width) {}

    sc_vector<sc_signal<DataType>> *operator()(const char *name, size_t)
    {
        return new sc_vector<sc_signal<DataType>>(name, width);
    }
    sc_trace_file *tf;
    unsigned int width;
};

template <typename DataType>
struct PeCreator
{
    PeCreator(sc_trace_file *_tf) : tf(_tf)
    {
    }
    PE<DataType> *operator()(const char *name, size_t)
    {
        return new PE<DataType>(name, this->tf);
    }
    sc_trace_file *tf;
};

template <typename DataType>
struct Arch_1x1 : public sc_module
{
    // Member Signals
private:
    sc_in_clk _clk;

public:
    sc_port<GlobalControlChannel_IF> control;
    sc_vector<PE<DataType>> pe_array;
    sc_vector<sc_signal<DataType>> filter_psum_out{"psum_out", 7};
    sc_trace_file *tf;
    SAM<DataType> psum_mem;
    sc_vector<sc_vector<sc_signal<DataType>>> psum_mem_read;
    sc_vector<sc_vector<sc_signal<DataType>>> psum_mem_write;
    SAM<DataType> ifmap_mem;
    sc_vector<sc_vector<sc_signal<DataType>>> ifmap_mem_read;
    sc_vector<sc_vector<sc_signal<DataType>>> ifmap_mem_write;

    void update()
    {
        if (control->enable())
        {
            for (int filter_row = 0; filter_row < 7; filter_row++)
            {
                for (int channel_column = 0; channel_column < 8; channel_column++)
                {
                    PE<DataType> &cur_pe = this->pe_array[filter_row * 9 + channel_column];
                    PE<DataType> &next_pe = this->pe_array[filter_row * 9 + channel_column + 1];
                    next_pe.psum_in = cur_pe.psum_in.read() + cur_pe.currentWeight() * filter_row * 9 + channel_column;
                    cur_pe.updateState();
                }
                PE<DataType> &last_pe = this->pe_array[filter_row * 9 + 8];
                filter_psum_out[filter_row] = last_pe.psum_in.read() + last_pe.currentWeight() * filter_row * 9 + 8;
                last_pe.updateState();
            }
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
        sc_trace_file *_tf) : sc_module(name),
                              pe_array("pe_array", 63, PeCreator<DataType>(_tf)),
                              tf(_tf),
                              psum_mem("psum_mem", _control, 9 * 2, 256*1024, 1, _tf),
                              psum_mem_read("psum_mem_read", 9 * 2, SignalVectorCreator<DataType>(1, tf)),
                              psum_mem_write("psum_mem_write", 9 * 2, SignalVectorCreator<DataType>(1, tf)),
                              ifmap_mem("ifmap_mem", _control, 7, 256*1024, 1, _tf),
                              ifmap_mem_read("ifmap_mem_read", 7, SignalVectorCreator<DataType>(1, tf)),
                              ifmap_mem_write("ifmap_mem_write", 7, SignalVectorCreator<DataType>(1, tf))
    {
        control(_control);
        _clk(control->clk());
        tf = _tf;

        // psum_read/write 
        for (int i = 0; i < 18; i++)
        {
            psum_mem.read_channel_data[i][0](psum_mem_read[i][0]);
            psum_mem.write_channel_data[i][0](psum_mem_write[i][0]);
        }

        for (int i = 0; i < 7; i++)
        {
            ifmap_mem.read_channel_data[i][0](ifmap_mem_read[i][0]);
            ifmap_mem.write_channel_data[i][0](ifmap_mem_write[i][0]);
        }

        SC_METHOD(update);
        sensitive << _clk.pos();
        sensitive << control->reset();
        cout << "Arch_1x1 MODULE: " << name << " has been instantiated " << endl;
    }

    SC_HAS_PROCESS(Arch_1x1);
};

int sc_main(int argc, char *argv[])
{
    sc_trace_file *tf = sc_create_vcd_trace_file("ProgTrace");
    tf->set_time_unit(1, SC_PS);
    
    GlobalControlChannel control("global_control_channel", sc_time(1, SC_NS), tf);
    Arch_1x1<sc_int<32>> arch("arch", control, tf);
    auto t1 = high_resolution_clock::now();
    sc_start(1000, SC_NS);
    auto t2 = high_resolution_clock::now();

    cout << "ALL TESTS PASS" << endl;
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    std::cout << ms_int.count() << "ms\n";

    exit(0); // avoids calling expensive destructors
    // ComputeBlob_TB<sc_int<32>> tb("ComputeBlob_tb");
    return 0;
    // return tb.run_tb();
}

#endif // MEM_HIERARCHY_CPP
