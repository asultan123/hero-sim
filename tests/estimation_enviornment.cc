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
#include <boost/program_options.hpp>

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

namespace po = boost::program_options;
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
struct Arch : public sc_module
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

    unsigned int dram_access_counter{0};
    int filter_count;
    int channel_count;
    int psum_mem_size;
    int ifmap_mem_size;

    void suspend_monitor()
    {
        while (1)
        {
            while (control->enable())
            {
                bool pes_suspended = true;
                for (auto &pe : pe_array)
                {
                    pes_suspended &= (pe.program.at(pe.prog_idx).state == DescriptorState::SUSPENDED);
                }
                bool ifmap_generators_suspended = true;
                for (auto &gen : ifmap_mem.generators)
                {
                    ifmap_generators_suspended &= (gen.currentDescriptor().state == DescriptorState::SUSPENDED);
                }
                bool psum_generators_suspended = true;
                for (auto &gen : psum_mem.generators)
                {
                    psum_generators_suspended &= (gen.currentDescriptor().state == DescriptorState::SUSPENDED);
                }
                if (pes_suspended && ifmap_generators_suspended && psum_generators_suspended)
                {
                    sc_stop();
                }
                wait();
            }
            wait();
        }
    }

    void update_1x1()
    {
        while (1)
        {
            while (control->enable())
            {
                for (int filter_row = 0; filter_row < filter_count; filter_row++)
                {
                    PE<DataType> &first_pe_in_row = this->pe_array[filter_row * channel_count];
                    first_pe_in_row.psum_in = psum_mem_read.at(filter_row + filter_count).at(0).read();
                }
                for (int filter_row = 0; filter_row < filter_count; filter_row++)
                {
                    for (int channel_column = 0; channel_column < channel_count - 1; channel_column++)
                    {

                        PE<DataType> &cur_pe = this->pe_array[filter_row * channel_count + channel_column];
                        PE<DataType> &next_pe = this->pe_array[filter_row * channel_count + channel_column + 1];
                        if (cur_pe.current_weight.read() != -1)
                        {
                            cur_pe.active_counter++;
                            next_pe.psum_in = cur_pe.compute(ifmap_mem_read[channel_column][0].read());
                        }
                        else
                        {
                            // bypass
                            cur_pe.inactive_counter++;
                            next_pe.psum_in = cur_pe.psum_in.read();
                        }
                        cur_pe.updateState();
                    }
                    PE<DataType> &last_pe = this->pe_array[filter_row * channel_count + channel_count - 1];

                    if (last_pe.current_weight.read() != -1)
                    {
                        last_pe.active_counter++;
                        psum_mem_write[filter_row][0] = last_pe.compute(ifmap_mem_read[channel_count - 1][0].read());
                    }
                    else
                    {
                        last_pe.inactive_counter++;
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
    Arch(
        sc_module_name name,
        GlobalControlChannel &_control,
        int filter_count,
        int channel_count,
        int psum_mem_size,
        int ifmap_mem_size,
        sc_trace_file *_tf) : sc_module(name),
                              pe_array("pe_array", filter_count * channel_count, PeCreator<DataType>(_tf)),
                              tf(_tf),
                              psum_mem("psum_mem", _control, filter_count * 2, psum_mem_size, 1, _tf),
                              psum_mem_read("psum_mem_read", filter_count * 2, SignalVectorCreator<DataType>(1, tf)),
                              psum_mem_write("psum_mem_write", filter_count * 2, SignalVectorCreator<DataType>(1, tf)),
                              ifmap_mem("ifmap_mem", _control, channel_count, ifmap_mem_size, 1, _tf),
                              ifmap_mem_read("ifmap_mem_read", channel_count, SignalVectorCreator<DataType>(1, tf)),
                              ifmap_mem_write("ifmap_mem_write", channel_count, SignalVectorCreator<DataType>(1, tf))
    {
        control(_control);
        _clk(control->clk());
        this->filter_count = filter_count;
        this->channel_count = channel_count;
        this->psum_mem_size = psum_mem_size;
        this->ifmap_mem_size = ifmap_mem_size;

        // for(auto& psum: this->filter_psum_out)
        // {
        //     sc_trace(tf, psum, psum.name());
        // }

        // psum_read/write
        for (int i = 0; i < filter_count * 2; i++)
        {
            psum_mem.read_channel_data[i][0](psum_mem_read[i][0]);
            psum_mem.write_channel_data[i][0](psum_mem_write[i][0]);
        }
        for (int i = 0; i < filter_count; i++)
        {
            psum_mem.channels[i].set_mode(MemoryChannelMode::WRITE);
            sc_trace(tf, psum_mem_write[i][0], (this->psum_mem_write[i][0].name()));
        }
        for (int i = filter_count; i < filter_count * 2; i++)
        {
            psum_mem.channels[i].set_mode(MemoryChannelMode::READ);
            sc_trace(tf, psum_mem_read[i][0], (this->psum_mem_read[i][0].name()));
        }

        for (int i = 0; i < channel_count; i++)
        {
            ifmap_mem.channels[i].set_mode(MemoryChannelMode::READ);
            ifmap_mem.read_channel_data[i][0](ifmap_mem_read[i][0]);
            ifmap_mem.write_channel_data[i][0](ifmap_mem_write[i][0]);
            sc_trace(tf, ifmap_mem_read[i][0], (this->ifmap_mem_read[i][0].name()));
        }

        SC_THREAD(update_1x1);
        sensitive << _clk.pos();
        sensitive << control->reset();

        SC_THREAD(suspend_monitor);
        sensitive << _clk.pos();
        sensitive << control->reset();
        cout << "Arch MODULE: " << name << " has been instantiated " << endl;
    }

    SC_HAS_PROCESS(Arch);
};

template <typename DataType>
void set_channel_modes(Arch<DataType> &arch)
{

    for (int i = 0; i < arch.filter_count; i++)
    {
        arch.psum_mem.channels[i].set_mode(MemoryChannelMode::WRITE);
    }
    for (int i = arch.filter_count; i < arch.filter_count * 2; i++)
    {
        arch.psum_mem.channels[i].set_mode(MemoryChannelMode::READ);
    }

    for (int i = 0; i < arch.channel_count; i++)
    {
        arch.ifmap_mem.channels[i].set_mode(MemoryChannelMode::READ);
    }
}

template <typename DataType>
xt::xarray<int> dram_load(Arch<DataType> &arch, int channel_in, int ifmap_h, int ifmap_w)
{
    auto input_size = ifmap_h * ifmap_w * channel_in;
    assert(input_size <= arch.ifmap_mem_size);

    xt::xarray<int> ifmap = xt::arange((int)1, input_size + 1);
    ifmap.reshape({channel_in, ifmap_h, ifmap_w});

    // cout << "IFMAP" << endl;
    // cout << ifmap << endl;

    for (int c = 0; c < channel_in; c++)
    {
        for (int i = 0; i < ifmap_h; i++)
        {
            for (int j = 0; j < ifmap_w; j++)
            {
                auto &mem_ptr = arch.ifmap_mem.mem.ram.at(c * (ifmap_h * ifmap_w) + i * ifmap_w + j).at(0);
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
xt::xarray<int> dram_store(Arch<DataType> &arch, int filter_out, int ofmap_h, int ofmap_w)
{
    auto output_size = ofmap_h * ofmap_w * filter_out;
    assert(output_size <= arch.psum_mem_size);
    xt::xarray<int> result = xt::zeros<int>({filter_out, ofmap_h, ofmap_w});
    for (int f = 0; f < filter_out; f++)
    {
        for (int i = 0; i < ofmap_h; i++)
        {
            for (int j = 0; j < ofmap_w; j++)
            {
                auto &mem_ptr = arch.psum_mem.mem.ram.at(f * (ofmap_h * ofmap_w) + i * ofmap_w + j).at(0);
                result(f, i, j) = mem_ptr.read();
                arch.dram_access_counter++;
                arch.psum_mem.mem.access_counter++;
            }
        }
    }
    cout << "Loaded dram contents from psum mem" << endl;
    return result;
}

template <typename DataType>
void generate_and_load_pe_program(Arch<DataType> &arch, int ifmap_h, int ifmap_w)
{
    int stream_size = ifmap_h * ifmap_w;
    int delay_offset = 1;
    for (int channel_column = 0; channel_column < arch.channel_count; channel_column++)
    {
        for (int filter_row = 0; filter_row < arch.filter_count; filter_row++)
        {
            PE<DataType> &cur_pe = arch.pe_array[filter_row * arch.channel_count + channel_column];
            vector<Descriptor_2D> program;
            program.push_back(Descriptor_2D::delay_inst(channel_column + delay_offset));
            program.push_back(Descriptor_2D::genhold_inst(0, stream_size, cur_pe.weights.size() - 1, 1));
            program.push_back(Descriptor_2D::suspend_inst());
            cur_pe.loadProgram(program);
        }
    }
}

template <typename DataType>
void generate_and_load_psum_program(Arch<DataType> &arch, xt::xarray<int> padded_weights, int ofmap_h, int ofmap_w)
{
    int verticle_tile_count = padded_weights.shape()[0] / arch.filter_count;
    int horizontal_tile_count = padded_weights.shape()[1] / arch.channel_count;

    int stream_size = ofmap_h * ofmap_w;

    xt::xarray<int> run_bitmap = xt::zeros<int>({verticle_tile_count, (int)arch.filter_count});
    for (auto filter_offset = 0; filter_offset < (int)padded_weights.shape()[0]; filter_offset += arch.filter_count)
    {
        for (auto channel_offset = 0; channel_offset < (int)padded_weights.shape()[1]; channel_offset += arch.channel_count)
        {
            auto tiled_view = xt::view(padded_weights, xt::range(filter_offset, filter_offset + arch.filter_count), xt::range(channel_offset, channel_offset + arch.channel_count));
            for (int filter = 0; filter < arch.filter_count; filter++)
            {
                int verticle_tile_idx = filter_offset / arch.filter_count;
                if (tiled_view(filter, 0) != -1)
                {
                    run_bitmap(verticle_tile_idx, filter) = 1;
                }
            }
        }
    }

    // cout << padded_weights << endl;
    // cout << run_bitmap << endl;

    for (int write_gen_idx = 0; write_gen_idx < arch.filter_count; write_gen_idx++)
    {
        vector<Descriptor_2D> program;

        program.push_back(Descriptor_2D::delay_inst(arch.channel_count + 1));
        for (int v = 0; v < verticle_tile_count; v++)
        {
            auto active = run_bitmap(v, write_gen_idx);
            if (active)
            {
                for (int h = 0; h < horizontal_tile_count; h++)
                {
                    program.push_back(Descriptor_2D::stream_inst(v * arch.filter_count * stream_size + write_gen_idx * stream_size, stream_size - 1, 0));
                }
            }
        }

        program.push_back(Descriptor_2D::suspend_inst());
        Descriptor_2D::make_sequential(program);
        arch.psum_mem.generators.at(write_gen_idx).loadProgram(program);
    }

    for (int read_gen_idx = arch.filter_count; read_gen_idx < arch.filter_count * 2; read_gen_idx++)
    {
        vector<Descriptor_2D> program;
        program.push_back(Descriptor_2D::delay_inst(3));

        for (int v = 0; v < verticle_tile_count; v++)
        {
            auto active = run_bitmap(v, (read_gen_idx - arch.filter_count));
            if (active)
            {
                program.push_back(Descriptor_2D::delay_inst(stream_size - 4 * (v == 0) - 1));
                for (int h = 1; h < horizontal_tile_count; h++)
                {
                    program.push_back(Descriptor_2D::stream_inst((v * arch.filter_count * stream_size) + (read_gen_idx - arch.filter_count) * stream_size, stream_size - 1, 0));
                }
            }
        }
        program.push_back(Descriptor_2D::suspend_inst());
        Descriptor_2D::make_sequential(program);
        arch.psum_mem.generators.at(read_gen_idx).loadProgram(program);
    }
}

template <typename DataType>
void generate_and_load_ifmap_in_program(Arch<DataType> &arch, xt::xarray<int> padded_weights, int ifmap_h, int ifmap_w)
{
    int verticle_tile_count = padded_weights.shape()[0] / arch.filter_count;
    int horizontal_tile_count = padded_weights.shape()[1] / arch.channel_count;

    xt::xarray<int> run_bitmap = xt::zeros<int>({verticle_tile_count, horizontal_tile_count, (int)arch.channel_count});
    for (auto filter_offset = 0; filter_offset < (int)padded_weights.shape()[0]; filter_offset += arch.filter_count)
    {
        for (auto channel_offset = 0; channel_offset < (int)padded_weights.shape()[1]; channel_offset += arch.channel_count)
        {
            auto tiled_view = xt::view(padded_weights, xt::range(filter_offset, filter_offset + arch.filter_count), xt::range(channel_offset, channel_offset + arch.channel_count));
            for (int channel = 0; channel < arch.channel_count; channel++)
            {
                int verticle_tile_idx = filter_offset / arch.filter_count;
                int horizontal_tile_idx = channel_offset / arch.channel_count;
                if (tiled_view(0, channel) != -1)
                {
                    run_bitmap(verticle_tile_idx, horizontal_tile_idx, channel) = 1;
                }
            }
        }
    }

    // cout << padded_weights << endl;

    // cout << run_bitmap << endl;

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
                int stream_start_idx = h * arch.channel_count * stream_size + ag_idx * stream_size;

                if (active)
                {
                    auto stream_inst = Descriptor_2D::stream_inst(stream_start_idx, stream_size - 1, 0);
                    program.push_back(stream_inst);
                }
                else
                {
                    auto delay_inst = Descriptor_2D::delay_inst(stream_size - 1);
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
tuple<xt::xarray<int>, xt::xarray<int>> generate_and_load_weights(Arch<DataType> &arch, int filter_out_dim, int channel_in_dim, int kernel, UnrollOrientation unroll_orientation)
{
    int kernel_size = kernel * kernel;
    xt::xarray<int> weights = xt::arange(1, channel_in_dim * filter_out_dim * kernel_size + 1);
    vector<vector<deque<int>>> pe_weights(arch.filter_count, vector<deque<int>>(arch.channel_count, deque<int>()));

    long unsigned int verticle_padding;
    long unsigned int horizontal_padding;

    switch (unroll_orientation)
    {
    case UnrollOrientation::HORIZONTAL:
    {
        weights.reshape({filter_out_dim, channel_in_dim * kernel_size});
        verticle_padding = ceil((float)filter_out_dim / arch.filter_count) * arch.filter_count - filter_out_dim;
        horizontal_padding = ceil((float)(channel_in_dim * kernel_size) / arch.channel_count) * arch.channel_count - (channel_in_dim * kernel_size);
        break;
    }
    default:
        cout << "INVALID ORIENTATION" << endl;
        exit(EXIT_FAILURE);
        break;
    }

    xt::xarray<int> padded_weights = xt::pad(weights, {{0, verticle_padding}, {0, horizontal_padding}}, xt::pad_mode::constant, PAD);

    // cout << padded_weights << endl;

    for (auto filter_offset = 0; filter_offset < (int)padded_weights.shape()[0]; filter_offset += arch.filter_count)
    {
        for (auto channel_offset = 0; channel_offset < (int)padded_weights.shape()[1]; channel_offset += arch.channel_count)
        {
            auto tiled_view = xt::view(padded_weights, xt::range(filter_offset, filter_offset + arch.filter_count), xt::range(channel_offset, channel_offset + arch.channel_count));

            for (auto i = 0; i < arch.filter_count; i++)
            {
                for (auto j = 0; j < arch.channel_count; j++)
                {
                    pe_weights[i][j].push_back(tiled_view(i, j));
                    arch.dram_access_counter++;
                }
            }
        }
    }

    for (int filter_row = 0; filter_row < arch.filter_count; filter_row++)
    {
        for (int channel_column = 0; channel_column < arch.channel_count; channel_column++)
        {
            auto &cur_pe = arch.pe_array[filter_row * arch.channel_count + channel_column];
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
    // cout << xt::adapt(weights.shape()) << endl;
    // ifmap.shape() = C*H*W
    assert(ifmap.shape().size() == 3);
    // cout << xt::adapt(ifmap.shape()) << endl;
    // cout << ifmap << endl;
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

bool validate_expected_output(xt::xarray<int> expected, xt::xarray<int> result)
{
    // cout << "EXPECTED RESULT" << endl;
    // cout << expected << endl;
    // cout << "ACTUAL RESULT" << endl;
    // cout << result << endl;
    return expected == result;
}

template <typename DataType>
void sim_and_get_results(int ifmap_h, int ifmap_w, int k, int c_in, int f_out, int filter_count, int channel_count)
{
    auto t1 = high_resolution_clock::now();

    int ofmap_h = (ifmap_h - k + 1);
    int ofmap_w = (ifmap_w - k + 1);
    int ifmap_mem_size = c_in * ifmap_h * ifmap_w;
    int psum_mem_size = f_out * ofmap_h * ofmap_w;

    xt::xarray<int> weights, padded_weights;

    xt::print_options::set_threshold(10000);
    xt::print_options::set_line_width(100);

    sc_trace_file *tf = sc_create_vcd_trace_file("Arch1x1");
    tf->set_time_unit(100, SC_PS);

    GlobalControlChannel control("global_control_channel", sc_time(1, SC_NS), tf);
    Arch<DataType> arch("arch", control, filter_count, channel_count, psum_mem_size, ifmap_mem_size, tf);

    unsigned long int start_cycle_time = sc_time_stamp().value();
    control.set_reset(true);
    sc_start(10, SC_NS);
    control.set_reset(false);
    sc_start(1, SC_NS);

    auto ifmap = dram_load(arch, c_in, ifmap_h, ifmap_w);
    // cout << ifmap << endl;

    set_channel_modes(arch);
    std::tie(weights, padded_weights) = generate_and_load_weights(arch, f_out, c_in, k, UnrollOrientation::HORIZONTAL);

    // cout << "PADDED WEIGHTS" << endl;
    // cout << padded_weights << endl;

    generate_and_load_pe_program(arch, ifmap_h, ifmap_w);
    generate_and_load_ifmap_in_program(arch, padded_weights, ifmap_h, ifmap_w);
    generate_and_load_psum_program(arch, padded_weights, ofmap_h, ofmap_w);

    control.set_program(true);
    sc_start(1, SC_NS);
    control.set_enable(true);
    control.set_program(false);
    sc_start();

    auto res = dram_store(arch, f_out, ofmap_h, ofmap_w);
    auto expected_ofmap = generate_expected_output(ifmap, weights);
    auto valid = validate_expected_output(expected_ofmap, res);
    unsigned long int end_cycle_time = sc_time_stamp().value();

    auto t2 = high_resolution_clock::now();
    auto sim_time = duration_cast<milliseconds>(t2 - t1);

    if (valid)
    {
        cout << "PASS" << endl;
        int weight_access = 0;
        xt::xarray<float> pe_utilization = xt::zeros<float>({1, (int)arch.pe_array.size()});
        int pe_idx = 0;
        for (auto &pe : arch.pe_array)
        {
            weight_access += pe.weight_access_counter;
            pe_utilization(0, pe_idx++) = (float)pe.active_counter / (float)(pe.active_counter + pe.inactive_counter);
        }
        float avg_util = xt::average(pe_utilization)(0);
        cout << std::left << std::setw(20) << "DRAM Access" << arch.dram_access_counter << endl;
        cout << std::left << std::setw(20) << "Weight Access" << weight_access << endl;
        cout << std::left << std::setw(20) << "Psum Access" << arch.psum_mem.mem.access_counter << endl;
        cout << std::left << std::setw(20) << "Ifmap Access" << arch.ifmap_mem.mem.access_counter << endl;
        cout << std::left << std::setw(20) << "Avg. Pe Util" << std::setprecision(2) << avg_util << endl;
        cout << std::left << std::setw(20) << "Latency in cycles" << end_cycle_time - start_cycle_time << endl;
        cout << std::left << std::setw(20) << "Simulated in " << sim_time.count() << "ms\n";
        exit(EXIT_SUCCESS); // avoids expensive de-alloc
    }
    else
    {
        cout << "FAIL" << endl;
    }
}

int sc_main(int argc, char *argv[])
{
    int ifmap_h = 10;
    int ifmap_w = 10;
    int k = 1;
    int c_in = 16;
    int f_out = 16;
    int filter_count = 7;
    int channel_count = 9;
    try
    {
        po::options_description config("Configuration");
        config.add_options()("help", "produce help message")("ifmap_h", po::value<int>(), "set input feature map width")("ifmap_w", po::value<int>(), "set input feature map height")("k", po::value<int>(), "set kernel size")("c_in", po::value<int>(), "set ifmap channel count")("f_out", po::value<int>(), "set weight filter count")("filter_count", po::value<int>(), "set arch width")("channel_count", po::value<int>(), "set arch height");

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, config), vm);
        po::notify(vm);

        if (vm.count("help"))
        {
            cout << config << "\n";
            return 0;
        }

        ifmap_h = (vm.count("ifmap_h")) ? vm["ifmap_h"].as<int>() : ifmap_h;
        ifmap_w = (vm.count("ifmap_w")) ? vm["ifmap_w"].as<int>() : ifmap_w;
        k = (vm.count("k")) ? vm["k"].as<int>() : k;
        c_in = (vm.count("c_in")) ? vm["c_in"].as<int>() : c_in;
        f_out = (vm.count("f_out")) ? vm["f_out"].as<int>() : f_out;
        filter_count = (vm.count("filter_count")) ? vm["filter_count"].as<int>() : filter_count;
        channel_count = (vm.count("channel_count")) ? vm["channel_count"].as<int>() : channel_count;

        if (ifmap_h <= 0 || ifmap_w <= 0 || k <= 0 || c_in <= 0 || f_out <= 0 || filter_count <= 0 || channel_count <= 0)
        {
            throw std::invalid_argument("all passed arguments must be positive");
        }

        if ((ifmap_h * ifmap_w) < 11)
        {
            throw std::invalid_argument("total ifmap sizes below 11 currently unsupported");
        }

        if (k > 1)
        {
            throw std::invalid_argument("kernel sizes greater than 1 currently unsupported");
        }
    }
    catch (std::exception &e)
    {
        cout << "error: " << e.what() << "\n";
        cout << "FAIL" << endl;

        return 1;
    }
    catch (...)
    {
        cout << "Exception of unknown type!\n";
        cout << "FAIL" << endl;

        return 1;
    }
    cout << std::left << "Simulating arch with config:" << endl;
    cout << endl;

    cout << std::left << std::setw(20) << "filter_count"  << filter_count << endl;;
    cout << std::left << std::setw(20) << "channel_count"  << channel_count << endl;;
    cout << endl;

    cout << std::left << "With layer config:" << endl;
    cout << endl;
    cout << std::left << std::setw(20) << "ifmap_h"  << ifmap_h << endl;
    cout << std::left << std::setw(20) << "ifmap_w" << ifmap_w << endl;
    cout << std::left << std::setw(20) << "k" << k << endl;
    cout << std::left << std::setw(20) << "c_in" << c_in << endl;
    cout << std::left << std::setw(20) << "f_out" << f_out << endl;

    sim_and_get_results<sc_int<32>>(ifmap_h, ifmap_w, k, c_in, f_out, filter_count, channel_count);

    return 0;
}

#endif // MEM_HIERARCHY_CPP
