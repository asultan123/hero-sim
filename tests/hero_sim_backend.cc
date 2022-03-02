#ifndef __ESTIMATION_ENVIORNMENT_CC
#define __ESTIMATION_ENVIORNMENT_CC

#include <systemc.h>
#include "descriptor_compiler.hh"
#include "hero.hh"
#include "layer_generation.hh"
#include "GlobalControl.hh"
#include <assert.h>
#include <iostream>
#include <string>
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
void dram_load(Hero::Arch<DataType> &arch, xt::xarray<int> ifmap, int channel_in, int ifmap_h, int ifmap_w)
{
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
}

template <typename DataType>
xt::xarray<int> dram_store(Hero::Arch<DataType> &arch, int filter_out, int ofmap_h, int ofmap_w)
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
void load_padded_weights_into_pes(Hero::Arch<DataType> &arch, xt::xarray<int> padded_weights)
{
    vector<vector<deque<int>>> pe_weights(arch.filter_count, vector<deque<int>>(arch.channel_count, deque<int>()));

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
    Hero::Arch<DataType> arch("arch", control, filter_count, channel_count, psum_mem_size, ifmap_mem_size, tf);

    unsigned long int start_cycle_time = sc_time_stamp().value();
    control.set_reset(true);
    sc_start(10, SC_NS);
    control.set_reset(false);
    sc_start(1, SC_NS);

    auto ifmap = LayerGeneration::generate_ifmap<DataType>(arch, c_in, ifmap_h, ifmap_w);
    dram_load(arch, ifmap, c_in, ifmap_h, ifmap_w);
    // cout << ifmap << endl;

    arch.set_channel_modes();
    weights = LayerGeneration::generate_weights<DataType>(f_out, c_in, k);
    padded_weights = LayerGeneration::pad_weights(arch, weights, f_out, c_in, k);

    load_padded_weights_into_pes(arch, padded_weights);

    // cout << "PADDED WEIGHTS" << endl;
    // cout << padded_weights << endl;

    GenerateDescriptors1x1::generate_and_load_pe_program(arch, ifmap_h, ifmap_w);
    GenerateDescriptors1x1::generate_and_load_ifmap_in_program(arch, padded_weights, ifmap_h, ifmap_w);
    GenerateDescriptors1x1::generate_and_load_psum_program(arch, padded_weights, ofmap_h, ofmap_w);

    control.set_program(true);
    sc_start(1, SC_NS);
    control.set_enable(true);
    control.set_program(false);
    sc_start();

    auto res = dram_store(arch, f_out, ofmap_h, ofmap_w);
    auto valid = LayerGeneration::validate_output_1x1(ifmap, weights, res);
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
        cout << std::left << std::setw(20) << "ALL TESTS PASS\n";
        exit(EXIT_SUCCESS); // avoids expensive de-alloc
    }
    else
    {
        cout << "FAIL" << endl;
    }
}

int sc_main(int argc, char *argv[])
{
    int ifmap_h = 32;
    int ifmap_w = 32;
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

    cout << std::left << std::setw(20) << "filter_count" << filter_count << endl;
    ;
    cout << std::left << std::setw(20) << "channel_count" << channel_count << endl;
    ;
    cout << endl;

    cout << std::left << "With layer config:" << endl;
    cout << endl;
    cout << std::left << std::setw(20) << "ifmap_h" << ifmap_h << endl;
    cout << std::left << std::setw(20) << "ifmap_w" << ifmap_w << endl;
    cout << std::left << std::setw(20) << "k" << k << endl;
    cout << std::left << std::setw(20) << "c_in" << c_in << endl;
    cout << std::left << std::setw(20) << "f_out" << f_out << endl;

    sim_and_get_results<sc_int<32>>(ifmap_h, ifmap_w, k, c_in, f_out, filter_count, channel_count);

    return 0;
}

#endif // MEM_HIERARCHY_CPP
