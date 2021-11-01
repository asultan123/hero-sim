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
#include <vector>
#include <assert.h>
#include <iomanip>
#include <cmath>
#include <deque>
#include <memory>
#include "AddressGenerator.hh"
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>

using std::cout;
using std::deque;
using std::endl;
using std::string;
using std::vector;
using std::chrono::duration;
using std::chrono::duration_cast;
using std::chrono::high_resolution_clock;
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


const long unsigned filter_count{7};
const long unsigned channel_count{9};
const long unsigned pe_count{filter_count * channel_count};

const long unsigned ifmap_mem_size{10*10*10};
const long unsigned psum_mem_size{10*10};
const long unsigned dram_access_cost{200};

template <typename DataType>
struct Arch_1x1 : public sc_module
{
    // Member Signals
private:
    sc_in_clk _clk;

public:
    sc_port<GlobalControlChannel_IF> control;
    sc_vector<PE<DataType>> pe_array;
    // sc_vector<sc_signal<DataType>> filter_psum_out{"filter_psum_out", filter_count};
    sc_trace_file *tf;
    SAM<DataType> psum_mem;
    sc_vector<sc_vector<sc_signal<DataType>>> psum_mem_read;

    sc_vector<sc_vector<sc_signal<DataType>>> psum_mem_write;
    SAM<DataType> ifmap_mem;
    sc_vector<sc_vector<sc_signal<DataType>>> ifmap_mem_read;
    sc_vector<sc_vector<sc_signal<DataType>>> ifmap_mem_write;
    vector<float> res;

    unsigned int dram_access_counter{0};
    unsigned int pe_mem_access_counter{0};

    void update()
    {
        while(1)
        {
            while (control->enable())
            {
                for (long unsigned filter_row = 0; filter_row < filter_count; filter_row++)
                {
                    PE<DataType> &first_pe_in_row = this->pe_array[filter_row * channel_count];
                    first_pe_in_row.psum_in = psum_mem_read[filter_row + filter_count][0].read();
                }
                for (long unsigned filter_row = 0; filter_row < filter_count; filter_row++)
                {
                    for (long unsigned channel_column = 0; channel_column < channel_count - 1; channel_column++)
                    {
                        PE<DataType> &cur_pe = this->pe_array[filter_row * channel_count + channel_column];
                        PE<DataType> &next_pe = this->pe_array[filter_row * channel_count + channel_column + 1];
                        next_pe.psum_in = cur_pe.compute(ifmap_mem_read[channel_column][0].read());
                        cur_pe.updateState();
                    }
                    PE<DataType> &last_pe = this->pe_array[filter_row * channel_count + channel_count - 1];
                    // filter_psum_out[filter_row] = last_pe.compute(ifmap_mem_read[channel_count - 1][0].read());
                    psum_mem_write[filter_row][0] = last_pe.compute(ifmap_mem_read[channel_count - 1][0].read());
                    last_pe.updateState();
                }
                wait();
            }
            wait();
        }
    }

    // Constructor
    Arch_1x1(
        sc_module_name name,
        GlobalControlChannel &_control,
        sc_trace_file *_tf) : sc_module(name),
                              pe_array("pe_array", pe_count, PeCreator<DataType>(_tf)),
                              tf(_tf),
                              psum_mem("psum_mem", _control, filter_count * 2, psum_mem_size, 1, _tf),
                              psum_mem_read("psum_mem_read", filter_count * 2, SignalVectorCreator<DataType>(1, tf)),
                              psum_mem_write("psum_mem_write", filter_count * 2, SignalVectorCreator<DataType>(1, tf)),
                              ifmap_mem("ifmap_mem", _control, channel_count, ifmap_mem_size, 1, _tf),
                              ifmap_mem_read("ifmap_mem_read", channel_count, SignalVectorCreator<DataType>(1, tf)),
                              ifmap_mem_write("ifmap_mem_write", channel_count, SignalVectorCreator<DataType>(1, tf)),
                              res(ifmap_mem_size)
    {
        control(_control);
        _clk(control->clk());

        // for(auto& psum: this->filter_psum_out)
        // {
        //     sc_trace(tf, psum, psum.name());
        // }

        // psum_read/write
        for (long unsigned int i = 0; i < filter_count * 2; i++)
        {
            psum_mem.read_channel_data[i][0](psum_mem_read[i][0]);
            psum_mem.write_channel_data[i][0](psum_mem_write[i][0]);
        }
        for (long unsigned int i = 0; i < filter_count; i++)
        {
            psum_mem.channels[i].set_mode(MemoryChannelMode::WRITE);
            sc_trace(tf, psum_mem_write[i][0], (this->psum_mem_write[i][0].name()));
        }
        for (long unsigned int i = filter_count; i < filter_count*2; i++)
        {
            psum_mem.channels[i].set_mode(MemoryChannelMode::READ);
            sc_trace(tf, psum_mem_read[i][0], (this->psum_mem_read[i][0].name()));
        }

        for (long unsigned int i = 0; i < channel_count; i++)
        {
            ifmap_mem.channels[i].set_mode(MemoryChannelMode::READ);
            ifmap_mem.read_channel_data[i][0](ifmap_mem_read[i][0]);
            ifmap_mem.write_channel_data[i][0](ifmap_mem_write[i][0]);
            sc_trace(tf, ifmap_mem_read[i][0], (this->ifmap_mem_read[i][0].name()));
        }

        SC_THREAD(update);
        sensitive << _clk.pos();
        sensitive << control->reset();
        cout << "Arch_1x1 MODULE: " << name << " has been instantiated " << endl;
    }

    SC_HAS_PROCESS(Arch_1x1);
};

template <typename DataType>
void set_channel_modes(Arch_1x1<DataType> &arch)
{

    for (long unsigned int i = 0; i < filter_count; i++)
    {
        arch.psum_mem.channels[i].set_mode(MemoryChannelMode::WRITE);
    }
    for (long unsigned int i = filter_count; i < filter_count*2; i++)
    {
        arch.psum_mem.channels[i].set_mode(MemoryChannelMode::READ);
    }

    for (long unsigned int i = 0; i < channel_count; i++)
    {
        arch.ifmap_mem.channels[i].set_mode(MemoryChannelMode::READ);
    }
}


template <typename DataType>
void dram_load(Arch_1x1<DataType> &arch, long unsigned int ifmap_w, long unsigned int ifmap_h, long unsigned int channel_in)
{
    auto input_size = ifmap_h * ifmap_h * channel_in;
    assert(input_size < ifmap_mem_size);
    for (long unsigned int c = 0; c < channel_in; c++)
    {
        for (long unsigned int i = 0; i < ifmap_h; i++)
        {
            for (long unsigned int j = 0; j < ifmap_w; j++)
            {
                auto &mem_ptr = arch.ifmap_mem.mem.ram[c * (ifmap_h * ifmap_w) + i * ifmap_w + j][0];
                mem_ptr.write(c * (ifmap_h * ifmap_w) + i * ifmap_w + j + 1);
                arch.dram_access_counter++;
                arch.ifmap_mem.mem.access_counter++;
            }
        }
    }
    sc_start(1, SC_NS);
    cout << "Loaded dram contents into ifmap mem" << endl;
}

template <typename DataType>
void dram_store(Arch_1x1<DataType> &arch, long unsigned int ofmap_w, long unsigned int ofmap_h, long unsigned int channel_out)
{
    auto output_size = ofmap_h * ofmap_h * channel_out;
    assert(output_size < ifmap_mem_size);
    for (int c = 0; c < channel_out; c++)
    {
        for (int i = 0; i < ofmap_h; i++)
        {
            for (int j = 0; j < ofmap_w; j++)
            {
                auto &mem_ptr = arch.ifmap_mem.mem.ram[c * (ofmap_h * ofmap_w) + i * ofmap_w + j][0];
                auto &res_ptr = arch.res[c * (ofmap_h * ofmap_w) + i * ofmap_w + j];
                mem_ptr.read();
                arch.dram_access_counter++;
                arch.psum_mem.mem.access_counter++;
            }
        }
    }
    sc_start(1, SC_NS);
    cout << "Loaded dram contents into ifmap mem" << endl;
}

void print_vec(string name, vector<int> &vec)
{
    int idx = 0;
    for (auto &i : vec)
    {
        cout << name << "[" << idx++ << "]" << std::right << std::setw(5) << i << endl;
    }
}

void print_vec_2d(string name, vector<int> &vec, int row_max, int col_max)
{
    cout << name << endl;
    for (int i = 0; i < row_max; i++)
    {
        for (int j = 0; j < col_max; j++)
        {
            cout << std::right << std::setw(5) << vec.at(i * col_max + j);
        }
        cout << endl;
    }
}

template <typename DataType>
void generate_and_load_weights(Arch_1x1<DataType> &arch, int channel_in_dim, int filter_out_dim, int kernel_size, string unroll_orientation)
{

    vector<vector<int>> weights(filter_out_dim, vector<int>(channel_in_dim*kernel_size, 0));
    int row = 0;
    int col = 0;

    int effective_filter_count = filter_count;
    int effective_channel_count = channel_count;
    int usable_filter_count = filter_count;
    int usable_channel_count = channel_count;

    if (unroll_orientation == "horizontal")
    {
        assert(channel_count >= (unsigned long int)kernel_size);
        effective_channel_count /= kernel_size;
        usable_channel_count = effective_channel_count * kernel_size;
    }
    else if (unroll_orientation == "verticle")
    {

        assert((unsigned long int)filter_count >= (unsigned long int)kernel_size);
        effective_filter_count /= kernel_size;
        usable_filter_count = effective_filter_count * kernel_size;
    }

    int filter_tile_count = ceil(filter_out_dim / (float)effective_filter_count);
    int channel_tile_count = ceil(channel_in_dim / (float)effective_channel_count);

    vector<vector<deque<int>>> pe_weights(filter_count, vector<deque<int>>(channel_count, deque<int>()));

    int weight_val = 1;
    for (int filter_tile = 0; filter_tile < filter_tile_count; filter_tile++)
    {
        for (int channel_tile = 0; channel_tile < channel_tile_count; channel_tile++)
        {

            if (unroll_orientation == "horizontal")
            {

                int filter_tile_boundary = (filter_tile == filter_tile_count - 1) ? filter_out_dim % usable_filter_count : usable_filter_count;

                int channel_tile_boundary;

                if(channel_tile == channel_tile_count - 1)
                {
                    if(channel_in_dim % effective_channel_count == 0)
                    {
                        channel_tile_boundary = kernel_size;
                    }
                    else
                    {
                        channel_tile_boundary = kernel_size * (channel_in_dim % effective_channel_count);
                    }
                }
                else
                {
                    channel_tile_boundary = usable_channel_count;
                }

                for (int filter_row = 0; filter_row < filter_tile_boundary; filter_row++)
                {
                    for (int channel_column = 0; channel_column < channel_tile_boundary; channel_column++)
                    {
                        weights[row][col] = weight_val;
                        col++;
                        if (col == channel_in_dim*kernel_size)
                        {
                            col = 0;
                            row++;
                        }
                        pe_weights[filter_row][channel_column].push_back(weight_val++);
                    }
                }
                for (unsigned long int filter_row = 0; filter_row < filter_count; filter_row++)
                {
                    for (unsigned long int channel_column = 0; channel_column < channel_count; channel_column++)
                    {
                        if(filter_row >= (unsigned long int)filter_tile_boundary || channel_column >= (unsigned long int)channel_tile_boundary)
                        {
                            pe_weights[filter_row][channel_column].push_back(-1);
                        }
                    }
                }
            }
            else if (unroll_orientation == "verticle")
            {
                int filter_tile_boundary;

                int channel_tile_boundary = (channel_tile == channel_tile_count - 1) ? (channel_in_dim % effective_channel_count) : usable_channel_count;

                if(filter_tile == filter_tile_count - 1)
                {
                    if(filter_out_dim % effective_filter_count == 0)
                    {
                        filter_tile_boundary = kernel_size;
                    }
                    else
                    {
                        filter_tile_boundary = kernel_size * (filter_out_dim % effective_filter_count);
                    }
                }
                else
                {
                    filter_tile_boundary = usable_filter_count;
                }

                for (int channel_column = 0; channel_column < channel_tile_boundary; channel_column++)
                {
                    for (int filter_row = 0; filter_row < filter_tile_boundary; filter_row++)
                    {
                        weights[row][col] = weight_val;
                        col++;
                        if (col == channel_in_dim*kernel_size)
                        {
                            col = 0;
                            row++;
                        }
                        pe_weights[filter_row][channel_column].push_back(weight_val++);
                    }
                }
                for (unsigned long int channel_column = 0; channel_column < channel_count; channel_column++)
                {
                    for (unsigned long int filter_row = 0; filter_row < filter_count; filter_row++)
                    {
                        if(filter_row >= (unsigned long int)filter_tile_boundary || channel_column >= (unsigned long int)channel_tile_boundary)
                        {
                            pe_weights[filter_row][channel_column].push_back(-1);
                        }
                    }
                }
            }
        }
    }

    for (unsigned long int filter_row = 0; filter_row < filter_count; filter_row++)
    {
        for (unsigned long int channel_column = 0; channel_column < channel_count; channel_column++)
        {
            auto &cur_pe = arch.pe_array[filter_row * channel_count + channel_column];
            vector<int> pe_weight_temp(pe_weights[filter_row][channel_column].begin(), pe_weights[filter_row][channel_column].end());
            cur_pe.loadWeights(pe_weight_temp);
        }
    }


    // cout << "weights" << endl;
    // for(auto i : weights)
    // {
    //     for(auto j : i)
    //     {
    //         cout << std::right << std::setw(5) << j << std::flush;
    //     }
    //     cout << endl;
    // }

    // cout << endl;

    // int slice_idx = 0;
    // bool all_empty;
    // do
    // {
    //     all_empty = true;
    //     cout << std::right << std::setw(14) << slice_idx++ <<"";
    //     for (long unsigned int channel_column = 0; channel_column < channel_count; channel_column++)
    //     {
    //         cout << std::right << std::setw(10) << channel_column << std::flush;
    //     }
    //     cout << endl;
    //     cout << std::right << std::setw(13) << "__________";
    //     for (long unsigned int channel_column = 0; channel_column < channel_count; channel_column++)
    //     {
    //         cout << std::right << std::setw(10) << "__________" << std::flush;
    //     }
    //     cout << endl;
    //     for (long unsigned int filter_row = 0; filter_row < filter_count; filter_row++)
    //     {
    //         cout << std::right << std::setw(10) << filter_row << "   |" << std::flush;
    //         for (long unsigned int channel_column = 0; channel_column < channel_count; channel_column++)
    //         {
    //             if (pe_weights[filter_row][channel_column].size() > 0)
    //             {
    //                 cout << std::right << std::setw(10) << /* " ["<< filter_row << "]"<< "[" << channel_column << "] " <<  */ pe_weights[filter_row][channel_column].front() << std::flush;
    //                 pe_weights[filter_row][channel_column].pop_front();
    //                 all_empty = false;
    //             }
    //         }
    //         cout << endl;
    //     }
    //     cout << endl;

    // } while (!all_empty);
    // cout << endl;
}

template <typename DataType>
void generate_and_load_pe_program(Arch_1x1<DataType> &arch)
{
    vector<Descriptor_2D> program{
        Descriptor_2D(
            /*next*/ 1,
            /*start*/ 0,
            /*state*/ DescriptorState::WAIT,
            /*x_count*/ 0,
            /*x_modify*/ 0,
            /*y_count*/ 0,
            /*y_modify*/ 0
        ),
        Descriptor_2D(
            /*next*/ 2,
            /*start*/ 0,
            /*state*/ DescriptorState::GENWAIT,
            /*x_count*/ 9,
            /*x_modify*/ 0,
            /*y_count*/ 10,
            /*y_modify*/ 1
        ),
        Descriptor_2D(
            /*next*/ 2,
            /*start*/ 0,
            /*state*/ DescriptorState::SUSPENDED,
            /*x_count*/ 0,
            /*x_modify*/ 0,
            /*y_count*/ 0,
            /*y_modify*/ 0
        ),
    };
    for (unsigned long int channel_column = 0; channel_column < channel_count; channel_column++)
    {
        for (unsigned long int filter_row = 0; filter_row < filter_count; filter_row++)
        {
            PE<DataType> &cur_pe = arch.pe_array[filter_row * channel_count + channel_column];
            // program[0].x_count_update(channel_column);
            cur_pe.loadProgram(program);
        }
    }
}

template <typename DataType>
void generate_and_load_psum_program(Arch_1x1<DataType> &arch)
{
    vector<Descriptor_2D> write_program{
        Descriptor_2D(
            /*next*/ 1,
            /*start*/ 0,
            /*state*/ DescriptorState::WAIT,
            /*x_count*/ 1,
            /*x_modify*/ 0,
            /*y_count*/ 0,
            /*y_modify*/ 0
        ),
        Descriptor_2D(
            /*next*/ 2,
            /*start*/ 0,
            /*state*/ DescriptorState::GENERATE,
            /*x_count*/ 10,
            /*x_modify*/ 1,
            /*y_count*/ 10,
            /*y_modify*/ -10
        ),
        Descriptor_2D(
            /*next*/ 2,
            /*start*/ 0,
            /*state*/ DescriptorState::SUSPENDED,
            /*x_count*/ 0,
            /*x_modify*/ 0,
            /*y_count*/ 0,
            /*y_modify*/ 0
        ),
    };

    for(int write_gen_idx = 0; write_gen_idx < filter_count; write_gen_idx++)
    {
        arch.psum_mem.generators.at(write_gen_idx).loadProgram(write_program);

    }

    vector<Descriptor_2D> read_program{
        Descriptor_2D(
            /*next*/ 1,
            /*start*/ 0,
            /*state*/ DescriptorState::WAIT,
            /*x_count*/ 2,
            /*x_modify*/ 0,
            /*y_count*/ 0,
            /*y_modify*/ 0
        ),
        Descriptor_2D(
            /*next*/ 2,
            /*start*/ 0,
            /*state*/ DescriptorState::GENERATE,
            /*x_count*/ 10,
            /*x_modify*/ 1,
            /*y_count*/ 10,
            /*y_modify*/ -10
        ),
        Descriptor_2D(
            /*next*/ 2,
            /*start*/ 0,
            /*state*/ DescriptorState::SUSPENDED,
            /*x_count*/ 0,
            /*x_modify*/ 0,
            /*y_count*/ 0,
            /*y_modify*/ 0
        ),
    };

    for(int read_gen_idx = filter_count; read_gen_idx < filter_count*2; read_gen_idx++)
    {
        arch.psum_mem.generators.at(read_gen_idx).loadProgram(read_program);
    }
}



template <typename DataType>
void generate_and_load_ifmap_in_program(Arch_1x1<DataType> &arch)
{
    vector<Descriptor_2D> program{
        Descriptor_2D(
            /*next*/ 1,
            /*start*/ 0,
            /*state*/ DescriptorState::WAIT,
            /*x_count*/ 0,
            /*x_modify*/ 0,
            /*y_count*/ 0,
            /*y_modify*/ 0
        ),
        Descriptor_2D(
            /*next*/ 2,
            /*start*/ 0,
            /*state*/ DescriptorState::GENERATE,
            /*x_count*/ 10,
            /*x_modify*/ 1,
            /*y_count*/ 10,
            /*y_modify*/ -10
        ),
        Descriptor_2D(
            /*next*/ 2,
            /*start*/ 0,
            /*state*/ DescriptorState::SUSPENDED,
            /*x_count*/ 0,
            /*x_modify*/ 0,
            /*y_count*/ 0,
            /*y_modify*/ 0
        ),
    };


    int channel_idx = 0;
    for(auto& ag: arch.ifmap_mem.generators)
    {
        program[1].start = channel_idx * (10*10);
        ag.loadProgram(program);
        channel_idx ++ ;
    }
}


template <typename DataType>
long sim_and_get_results()
{
    sc_trace_file *tf = sc_create_vcd_trace_file("Arch1x1");
    tf->set_time_unit(100, SC_PS);

    GlobalControlChannel control("global_control_channel", sc_time(1, SC_NS), tf);
    Arch_1x1<DataType> arch("arch", control, tf);

    auto t1 = high_resolution_clock::now();
    control.set_reset(true);
    sc_start(10, SC_NS);
    control.set_reset(false);
    sc_start(10, SC_NS);
    dram_load(arch, ifmap_mem_size, 1, 1);
    set_channel_modes(arch);
    generate_and_load_weights(arch, 16, 16, 9, "horizontal");
    generate_and_load_pe_program(arch);
    generate_and_load_ifmap_in_program(arch);
    generate_and_load_psum_program(arch);

    control.set_program(true);
    sc_start(1, SC_NS);
    control.set_enable(true);
    control.set_program(false);
    sc_start(10000, SC_NS);
    auto t2 = high_resolution_clock::now();
    auto ms_int = duration_cast<milliseconds>(t2 - t1);
    return ms_int.count();
}

int xtensor_test()
{
    xt::xarray<double> arr1
      {{1.0, 2.0, 3.0},
       {2.0, 5.0, 7.0},
       {2.0, 5.0, 7.0}};

    xt::xarray<double> arr2
      {5.0, 6.0, 7.0};

    xt::xarray<double> res = xt::view(arr1, 1) + arr2;

    std::cout << res << std::endl;

    return 0;
}

int sc_main(int argc, char *argv[])
{

    // auto sim_time = sim_and_get_results<sc_int<32>>();
    // std::cout << sim_time << "ms\n";

    xtensor_test();

    cout << "ALL TESTS PASS" << endl;


    exit(EXIT_SUCCESS); // avoids expensive de-alloc
}

#endif // MEM_HIERARCHY_CPP
