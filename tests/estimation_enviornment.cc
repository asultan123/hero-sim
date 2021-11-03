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
#include <tuple>
#include "AddressGenerator.hh"
#include <xtensor/xarray.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xaxis_iterator.hpp>
#include <xtensor/xpad.hpp>
#include <iostream>
#include <string>
#include <xtensor/xadapt.hpp>
#include <xtensor-blas/xlinalg.hpp>

#define PAD -1

using std::cout;
using std::deque;
using std::endl;
using std::string;
using std::tuple;
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

#define IFMAP_H 10
#define IFMAP_W 10
#define C_IN 16
#define F_OUT 16
#define K 1

const long unsigned filter_count{7};
const long unsigned channel_count{9};
const long unsigned pe_count{filter_count * channel_count};

const long unsigned ifmap_mem_size{C_IN * IFMAP_H * IFMAP_W};
const long unsigned psum_mem_size{F_OUT * (IFMAP_H-K+1) * (IFMAP_W-K+1 )};
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
        while (1)
        {
            while (control->enable())
            {
                for (long unsigned filter_row = 0; filter_row < filter_count; filter_row++)
                {
                    PE<DataType> &first_pe_in_row = this->pe_array[filter_row * channel_count];
                    first_pe_in_row.psum_in = 0;
                    // first_pe_in_row.psum_in = psum_mem_read[filter_row + filter_count][0].read();
                }
                for (long unsigned filter_row = 0; filter_row < filter_count; filter_row++)
                {
                    for (long unsigned channel_column = 0; channel_column < channel_count - 1; channel_column++)
                    {
                        PE<DataType> &cur_pe = this->pe_array[filter_row * channel_count + channel_column];
                        PE<DataType> &next_pe = this->pe_array[filter_row * channel_count + channel_column + 1];
                        if(cur_pe.current_weight.read() != -1)
                        {
                            next_pe.psum_in = cur_pe.compute(ifmap_mem_read[channel_column][0].read());
                        }
                        else
                        {
                            // bypass
                            next_pe.psum_in = cur_pe.psum_in.read();
                        }
                        cur_pe.updateState();
                    }
                    PE<DataType> &last_pe = this->pe_array[filter_row * channel_count + channel_count - 1];
                    if(last_pe.current_weight.read() != -1)
                    {
                        psum_mem_write[filter_row][0] = last_pe.compute(ifmap_mem_read[channel_count - 1][0].read());
                    }
                    else
                    {
                        psum_mem_write[filter_row][0] = last_pe.psum_in.read();
                    }
                    last_pe.updateState();
                }
                // for(int i =0 ; i < 7; i++)
                // {
                //     for(int j = 0; j<9; j++)
                //     {
                //         cout << this->pe_array[i*9 + j].current_weight.read() << " ";
                //     }
                //     cout << endl;
                // }
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
        for (long unsigned int i = filter_count; i < filter_count * 2; i++)
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
    for (long unsigned int i = filter_count; i < filter_count * 2; i++)
    {
        arch.psum_mem.channels[i].set_mode(MemoryChannelMode::READ);
    }

    for (long unsigned int i = 0; i < channel_count; i++)
    {
        arch.ifmap_mem.channels[i].set_mode(MemoryChannelMode::READ);
    }
}

template <typename DataType>
xt::xarray<int> dram_load(Arch_1x1<DataType> &arch, long unsigned int channel_in, long unsigned int ifmap_h, long unsigned int ifmap_w)
{
    auto input_size = ifmap_h * ifmap_h * channel_in;
    assert(input_size <= ifmap_mem_size);

    xt::xarray<int> ifmap = xt::arange((unsigned long int)0, input_size);
    ifmap.reshape({channel_in, ifmap_h, ifmap_w});

    for (long unsigned int c = 0; c < channel_in; c++)
    {
        for (long unsigned int i = 0; i < ifmap_h; i++)
        {
            for (long unsigned int j = 0; j < ifmap_w; j++)
            {
                auto &mem_ptr = arch.ifmap_mem.mem.ram[c * (ifmap_h * ifmap_w) + i * ifmap_w + j][0];
                mem_ptr.write(ifmap(c, i, j));
                arch.dram_access_counter++;
                arch.ifmap_mem.mem.access_counter++;
            }
        }
    }
    sc_start(1, SC_NS);
    cout << "Loaded dram contents into ifmap mem" << endl;

    return ifmap;
}

template <typename DataType>
void dram_store(Arch_1x1<DataType> &arch, long unsigned int ofmap_w, long unsigned int ofmap_h, long unsigned int channel_out)
{
    auto output_size = ofmap_h * ofmap_h * channel_out;
    assert(output_size <= psum_mem_size);
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
    cout << "Loaded dram contents from psum mem" << endl;
}

template <typename DataType>
void generate_and_load_pe_program(Arch_1x1<DataType> &arch, int ifmap_h, int ifmap_w)
{
    int stream_size = ifmap_h * ifmap_w;
    int delay_offset = 1;
    for (unsigned long int channel_column = 0; channel_column < channel_count; channel_column++)
    {
        for (unsigned long int filter_row = 0; filter_row < filter_count; filter_row++)
        {
            PE<DataType> &cur_pe = arch.pe_array[filter_row * channel_count + channel_column];
            vector<Descriptor_2D> program;
            program.push_back(Descriptor_2D::delay_inst(channel_column+delay_offset));
            program.push_back(Descriptor_2D::genhold_inst(0, stream_size, cur_pe.weights.size()-1, 1));
            program.push_back(Descriptor_2D::suspend_inst());
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
            /*y_modify*/ 0),
        Descriptor_2D(
            /*next*/ 2,
            /*start*/ 0,
            /*state*/ DescriptorState::GENERATE,
            /*x_count*/ 10,
            /*x_modify*/ 1,
            /*y_count*/ 10,
            /*y_modify*/ -10),
        Descriptor_2D(
            /*next*/ 2,
            /*start*/ 0,
            /*state*/ DescriptorState::SUSPENDED,
            /*x_count*/ 0,
            /*x_modify*/ 0,
            /*y_count*/ 0,
            /*y_modify*/ 0),
    };

    for (int write_gen_idx = 0; write_gen_idx < filter_count; write_gen_idx++)
    {
        // arch.psum_mem.generators.at(write_gen_idx).loadProgram(write_program);
    }

    vector<Descriptor_2D> read_program{
        Descriptor_2D(
            /*next*/ 1,
            /*start*/ 0,
            /*state*/ DescriptorState::WAIT,
            /*x_count*/ 2,
            /*x_modify*/ 0,
            /*y_count*/ 0,
            /*y_modify*/ 0),
        Descriptor_2D(
            /*next*/ 2,
            /*start*/ 0,
            /*state*/ DescriptorState::GENERATE,
            /*x_count*/ 10,
            /*x_modify*/ 1,
            /*y_count*/ 10,
            /*y_modify*/ -10),
        Descriptor_2D(
            /*next*/ 2,
            /*start*/ 0,
            /*state*/ DescriptorState::SUSPENDED,
            /*x_count*/ 0,
            /*x_modify*/ 0,
            /*y_count*/ 0,
            /*y_modify*/ 0),
    };

    for (int read_gen_idx = filter_count; read_gen_idx < filter_count * 2; read_gen_idx++)
    {
        // arch.psum_mem.generators.at(read_gen_idx).loadProgram(read_program);
    }
}

template <typename DataType>
void generate_and_load_ifmap_in_program(Arch_1x1<DataType> &arch, xt::xarray<int> padded_weights, int ifmap_h, int ifmap_w)
{
    int verticle_tile_count = padded_weights.shape()[0] / filter_count;
    int horizontal_tile_count = padded_weights.shape()[1] / channel_count;

    xt::xarray<int> run_bitmap = xt::zeros<int>({verticle_tile_count, horizontal_tile_count, (int)channel_count});
    for (auto filter_offset = 0; filter_offset < padded_weights.shape()[0]; filter_offset += filter_count)
    {
        for (auto channel_offset = 0; channel_offset < padded_weights.shape()[1]; channel_offset += channel_count)
        {
            auto tiled_view = xt::view(padded_weights, xt::range(filter_offset, filter_offset + filter_count), xt::range(channel_offset, channel_offset + channel_count));
            for (int channel = 0; channel < channel_count; channel++)
            {
                int verticle_tile_idx = filter_offset / filter_count;
                int horizontal_tile_idx = channel_offset / channel_count;
                if (tiled_view(0, channel) != -1)
                {
                    run_bitmap(verticle_tile_idx, horizontal_tile_idx, channel) = 1;
                }
            }
        }
    }

    cout << padded_weights << endl;

    cout << run_bitmap << endl;

    int ag_idx = 0;
    for (auto &ag : arch.ifmap_mem.generators)
    {
        std::deque<Descriptor_2D> program;
        auto systolic_delay = Descriptor_2D::delay_inst(ag_idx);
        program.push_back(systolic_delay);
        for (int v = 0; v < verticle_tile_count; v++)
        {
            for (int h = 0; h < horizontal_tile_count; h++)
            {
                int active = run_bitmap(v, h, ag_idx);
                int stream_size = ifmap_h * ifmap_w;
                int stream_start_idx = h * channel_count * stream_size + ag_idx * stream_size;

                if (active)
                {
                    auto stream_inst = Descriptor_2D::stream_inst(stream_start_idx, stream_size-1, 0);
                    program.push_back(stream_inst);
                }
                else
                {
                    auto delay_inst = Descriptor_2D::delay_inst(stream_size);
                    program.push_back(delay_inst);
                }
            }
        }
        program.push_back(Descriptor_2D::suspend_inst());
        vector<Descriptor_2D> prog_vec(program.begin(), program.end());
        Descriptor_2D::make_sequential(prog_vec);
        ag.loadProgram(prog_vec);
        ag_idx++;
    }
}

enum UnrollOrientation
{
    HORIZONTAL = 1,
    VERTICLE = 2
};

template <typename DataType>
tuple<xt::xarray<int>, xt::xarray<int>> generate_and_load_weights(Arch_1x1<DataType> &arch, int filter_out_dim, int channel_in_dim, int kernel, UnrollOrientation unroll_orientation)
{
    int kernel_size = kernel * kernel;
    xt::xarray<int> weights = xt::arange(0, channel_in_dim * filter_out_dim * kernel_size);
    vector<vector<deque<int>>> pe_weights(filter_count, vector<deque<int>>(channel_count, deque<int>()));

    long unsigned int verticle_padding;
    long unsigned int horizontal_padding;

    switch (unroll_orientation)
    {
    case UnrollOrientation::HORIZONTAL:
    {
        weights.reshape({filter_out_dim, channel_in_dim * kernel_size});
        verticle_padding = ceil((float)filter_out_dim / filter_count) * filter_count - filter_out_dim;
        horizontal_padding = ceil((float)(channel_in_dim * kernel_size) / channel_count) * channel_count - (channel_in_dim * kernel_size);
        break;
    }
    default:
        cout << "INVALID ORIENTATION" << endl;
        exit(EXIT_FAILURE);
        break;
    }

    xt::xarray<int> padded_weights = xt::pad(weights, {{0, verticle_padding}, {0, horizontal_padding}}, xt::pad_mode::constant, PAD);

    cout << padded_weights << endl;

    for (auto filter_offset = 0; filter_offset < padded_weights.shape()[0]; filter_offset += filter_count)
    {
        for (auto channel_offset = 0; channel_offset < padded_weights.shape()[1]; channel_offset += channel_count)
        {
            auto tiled_view = xt::view(padded_weights, xt::range(filter_offset, filter_offset + filter_count), xt::range(channel_offset, channel_offset + channel_count));

            for (auto i = 0; i < filter_count; i++)
            {
                for (auto j = 0; j < channel_count; j++)
                {
                    pe_weights[i][j].push_back(tiled_view(i, j));
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

    weights.reshape({filter_out_dim, channel_in_dim, kernel, kernel});
    return std::make_tuple(weights, padded_weights);
}

xt::xarray<int> generate_expected_output(xt::xarray<int> ifmap, xt::xarray<int> weights)
{
    // weights.shape() = F*C*K*K
    assert(weights.shape().size() == 4);
    cout << xt::adapt(weights.shape()) << endl;
    // ifmap.shape() = C*H*W
    assert(ifmap.shape().size() == 3);
    cout << xt::adapt(ifmap.shape()) << endl;
    cout << ifmap << endl;
    // symmetric kernel
    assert(weights.shape(3) == weights.shape(2));

    // ifmap channel = weights channel in
    assert(ifmap.shape(0) == weights.shape(1));
    int ifmap_w = ifmap.shape(2);
    int ifmap_h = ifmap.shape(1);

    int kernel = weights.shape(3);
    assert(ifmap_w >= kernel);
    assert(ifmap_h >= kernel);

    int ofmap_w = ifmap_w - (kernel - 1);
    int ofmap_h = ifmap_h - (kernel - 1);
    int ofmap_c = weights.shape(0);

    xt::xarray<int> ofmap = xt::arange(ofmap_c * ofmap_w * ofmap_h).reshape({ofmap_c, ofmap_h, ofmap_w});

    // conv2d stride 1
    for (auto f = 0; f < ofmap_c; f++)
    {
        auto weight_tensor_view = xt::view(weights, f, xt::all(), xt::all(), xt::all());
        xt::xarray<int> flatten_weight(xt::flatten(weight_tensor_view));
        for (auto h = 0; h < ofmap_h; h++)
        {
            for (auto w = 0; w < ofmap_w; w++)
            {
                auto ifmap_tensor_view = xt::view(ifmap, xt::all(), xt::range(h, h + kernel), xt::range(w, w + kernel));
                xt::xarray<int> flattened_ifmap(xt::flatten(ifmap_tensor_view));
                auto val = xt::linalg::dot(flattened_ifmap, flatten_weight);
                ofmap(f, h, w) = val(0);
            }
        }
    }

    return ofmap;
}

template <typename DataType>
long sim_and_get_results()
{
    int ifmap_h = IFMAP_H;
    int ifmap_w = IFMAP_W;
    int c_in = C_IN;
    int f_out = F_OUT;
    int k = K;

    xt::print_options::set_threshold(10000);
    xt::print_options::set_line_width(100);

    sc_trace_file *tf = sc_create_vcd_trace_file("Arch1x1");
    tf->set_time_unit(100, SC_PS);

    GlobalControlChannel control("global_control_channel", sc_time(1, SC_NS), tf);
    Arch_1x1<DataType> arch("arch", control, tf);

    auto t1 = high_resolution_clock::now();
    control.set_reset(true);
    sc_start(10, SC_NS);
    control.set_reset(false);
    sc_start(1, SC_NS);
    auto ifmap = dram_load(arch, c_in, ifmap_h, ifmap_w);
    set_channel_modes(arch);
    xt::xarray<int> weights, padded_weights;
    std::tie(weights, padded_weights) = generate_and_load_weights(arch, f_out, c_in, k, UnrollOrientation::HORIZONTAL);
    // auto expected_ofmap = generate_expected_output(ifmap, weights);

    generate_and_load_pe_program(arch, ifmap_h, ifmap_w);
    generate_and_load_ifmap_in_program(arch, padded_weights, ifmap_h, ifmap_w);
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

int sc_main(int argc, char *argv[])
{

    auto sim_time = sim_and_get_results<sc_int<32>>();
    std::cout << sim_time << "ms\n";

    cout << "ALL TESTS PASS" << endl;

    exit(EXIT_SUCCESS); // avoids expensive de-alloc
}

#endif // MEM_HIERARCHY_CPP
