#ifndef __ESTIMATION_ENVIORNMENT_CC
#define __ESTIMATION_ENVIORNMENT_CC

#include "../hero-sim-proto/result.pb.h"
#include "AddressGenerator.hh"
#include "GlobalControl.hh"
#include "ProcEngine.hh"
#include "SAM.hh"
#include "descriptor_compiler.hh"
#include "hero.hh"
#include "layer_generation.hh"
#include <assert.h>
#include <boost/program_options.hpp>
#include <chrono>
#include <cmath>
#include <deque>
#include <fmt/format.h>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <systemc.h>
#include <tuple>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xaxis_iterator.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xview.hpp>

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
void dram_load_ifmap(Hero::Arch<DataType> &arch, xt::xarray<int> ifmap, int channel_in, int ifmap_h, int ifmap_w)
{
    // TODO #42
    for (int c = 0; c < channel_in; c++)
    {
        for (int i = 0; i < ifmap_h; i++)
        {
            for (int j = 0; j < ifmap_w; j++)
            {
                auto &mem_ptr = arch.ifmap_mem.mem.ram.at(c * (ifmap_h * ifmap_w) + i * ifmap_w + j).at(0);
                mem_ptr = (ifmap(c, i, j));
                arch.dram_access_counter++;
                arch.ifmap_mem.mem.access_counter++;
            }
        }
    }
    sc_start(1, SC_NS);
    cout << "Loaded dram contents into ifmap mem" << endl;
}

template <typename DataType> void dram_sim_load_bias(Hero::Arch<DataType> &arch)
{
    for (auto &_ : arch.psum_mem.mem.ram)
    {
        arch.dram_access_counter++;
        arch.psum_mem.mem.access_counter++;
    }

    sc_start(1, SC_NS);
    cout << "Loaded bias into psum mem" << endl;
}

template <typename DataType>
xt::xarray<DataType> dram_store(Hero::Arch<DataType> &arch, int filter_out, int ofmap_h, int ofmap_w)
{
    auto output_size = ofmap_h * ofmap_w * filter_out;
    assert(output_size <= arch.psum_mem_size);
    xt::xarray<DataType> result = xt::zeros<int>({filter_out, ofmap_h, ofmap_w});
    for (int f = 0; f < filter_out; f++)
    {
        for (int i = 0; i < ofmap_h; i++)
        {
            for (int j = 0; j < ofmap_w; j++)
            {
                auto &mem_ptr = arch.psum_mem.mem.ram.at(f * (ofmap_h * ofmap_w) + i * ofmap_w + j).at(0);
                result(f, i, j) = mem_ptr;
                arch.dram_access_counter++;
                arch.psum_mem.mem.access_counter++;
            }
        }
    }
    cout << "Loaded dram contents from psum mem" << endl;
    return result;
}

template <typename DataType>
xt::xarray<DataType> dram_store_with_filtering(Hero::Arch<DataType> &arch, int filter_out, int ifmap_h, int ifmap_w,
                                               int ofmap_h, int ofmap_w)
{
    auto output_size = ofmap_h * ofmap_w * filter_out;
    assert(output_size <= arch.psum_mem_size);
    xt::xarray<DataType> result = xt::zeros<int>({filter_out, ofmap_h, ofmap_w});
    for (int f = 0; f < filter_out; f++)
    {
        for (int i = 0; i < ofmap_h; i++)
        {
            for (int j = 2; j < ifmap_w; j++)
            {
                auto &mem_ptr = arch.psum_mem.mem.ram.at(f * (ofmap_h * ifmap_w) + i * ifmap_w + j).at(0);
                result(f, i, j - 2) = mem_ptr;
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
        for (auto channel_offset = 0; channel_offset < (int)padded_weights.shape()[1];
             channel_offset += arch.channel_count)
        {
            auto tiled_view = xt::view(padded_weights, xt::range(filter_offset, filter_offset + arch.filter_count),
                                       xt::range(channel_offset, channel_offset + arch.channel_count));

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
            vector<int> pe_weight_temp(pe_weights[filter_row][channel_column].begin(),
                                       pe_weights[filter_row][channel_column].end());
            cur_pe.loadWeights(pe_weight_temp);
        }
    }
}

template <typename DataType>
void sim_and_get_results(int ifmap_h, int ifmap_w, int k, int c_in, int f_out, int filter_count, int channel_count,
                         Hero::OperationMode op_mode, bool result_as_protobuf, bool sim_bias)
{
    auto t1 = high_resolution_clock::now();

    int ofmap_h = (ifmap_h - k + 1);
    int ofmap_w = (ifmap_w - k + 1);
    int ifmap_mem_size = c_in * ifmap_h * ifmap_w;
    int psum_mem_size;

    if (op_mode == Hero::OperationMode::RUN_1x1)
    {
        psum_mem_size = f_out * ofmap_h * ofmap_w;
    }
    else if (op_mode == Hero::OperationMode::RUN_3x3)
    {
        // TODO: #46
        psum_mem_size = f_out * ofmap_h * ifmap_w + 2;
    }
    else
    {
        throw "Invalid Accelerator Operation Mode";
    }

    xt::xarray<int> weights, padded_weights;

    xt::print_options::set_threshold(10000);
    xt::print_options::set_line_width(100);

#ifdef DEBUG
    sc_trace_file *tf = sc_create_vcd_trace_file("Arch1x1");
    tf->set_time_unit(10, SC_PS);
#else
    sc_trace_file *tf = nullptr;
#endif

    GlobalControlChannel control("global_control_channel", sc_time(1, SC_NS), tf);
    Hero::Arch<DataType> arch("arch", control, filter_count, channel_count, psum_mem_size, ifmap_mem_size, tf, op_mode);

    fmt::print("Instantiated HERO Arch\n");

    auto start_cycle_time = sc_time_stamp();
    control.set_reset(true);
    sc_start(10, SC_NS);
    control.set_reset(false);
    sc_start(1, SC_NS);

    auto ifmap = LayerGeneration::generate_ifmap<DataType>(arch, c_in, ifmap_h, ifmap_w);

    dram_load_ifmap(arch, ifmap, c_in, ifmap_h, ifmap_w);

    if (sim_bias)
    {
        dram_sim_load_bias(arch);
    }

    weights = LayerGeneration::generate_weights<DataType>(f_out, c_in, k);

    padded_weights = LayerGeneration::pad_weights(arch, weights, f_out, c_in, k);

    load_padded_weights_into_pes(arch, padded_weights);

    GenerateDescriptors::generate_and_load_arch_descriptors(arch, ifmap_h, ifmap_w, padded_weights, ofmap_h, ofmap_w);

    control.set_program(true);
    arch.set_channel_modes();
    sc_start(1, SC_NS);
    control.set_enable(true);
    control.set_program(false);

    try
    {
        sc_start();
    }
    catch (const std::exception &e)
    {
        std::cerr << e.what() << '\n';
        sc_close_vcd_trace_file(tf);
    }

    xt::xarray<DataType> arch_output;
    if (op_mode == Hero::OperationMode::RUN_1x1)
    {
        arch_output = dram_store(arch, f_out, ofmap_h, ofmap_w);
    }
    else if (op_mode == Hero::OperationMode::RUN_3x3)
    {
        arch_output = dram_store_with_filtering(arch, f_out, ifmap_h, ifmap_w, ofmap_h, ofmap_w);
    }
    else
    {
        throw "Invalid Accelerator Operation Mode";
    }

    fmt::print("Validating output\n");
    bool valid = LayerGeneration::validate_output(ifmap, weights, arch_output);
    auto end_cycle_time = sc_time_stamp();

    auto t2 = high_resolution_clock::now();
    auto sim_time = duration_cast<milliseconds>(t2 - t1);

    message::Result res;

    if (valid)
    {
        cout << "PASS" << endl;
        uint64_t weight_access = 0;
        xt::xarray<float> pe_utilization = xt::zeros<float>({1, (int)arch.pe_array.size()});
        int pe_idx = 0;
        uint64_t total_macs = 0;
        for (auto &pe : arch.pe_array)
        {
            weight_access += pe.weight_access_counter;
            total_macs += pe.active_counter;
            pe_utilization(0, pe_idx++) = (float)pe.active_counter / (float)(pe.active_counter + pe.inactive_counter);
        }
        float avg_util = xt::average(pe_utilization)(0);
        auto latency_in_cycles = end_cycle_time - start_cycle_time;
        cout << std::left << std::setw(20) << "DRAM Access" << arch.dram_access_counter << endl;
        cout << std::left << std::setw(20) << "Weight Access" << weight_access << endl;
        cout << std::left << std::setw(20) << "Psum Access" << arch.psum_mem.mem.access_counter << endl;
        cout << std::left << std::setw(20) << "Ifmap Access" << arch.ifmap_mem.mem.access_counter << endl;
        cout << std::left << std::setw(20) << "Avg. Pe Util" << std::setprecision(2) << avg_util << endl;
        cout << std::left << std::setw(20) << "Latency in cycles" << latency_in_cycles.value() / 1000 << endl;
        cout << std::left << std::setw(20) << "MACs Performed" << total_macs << endl;
        cout << std::left << std::setw(20) << "Simulated in " << sim_time.count() << "ms\n";
        cout << std::left << std::setw(20) << "ALL TESTS PASS\n";

        res.set_valid("PASS");
        res.set_dram_access(arch.dram_access_counter);
        res.set_weight_access(weight_access);
        res.set_ifmap_access(arch.psum_mem.mem.access_counter);
        res.set_psum_access(arch.ifmap_mem.mem.access_counter);
        res.set_avg_util(avg_util);
        res.set_latency(latency_in_cycles.value() / 1000);
        res.set_macs(total_macs);
        res.set_sim_time(sim_time.count());
    }
    else
    {
        cout << "FAIL" << endl;
        res.set_valid("FAIL");
    }

    if (result_as_protobuf)
    {
        std::string output_string = res.SerializeAsString();
        cerr << output_string;
    }

    exit(EXIT_SUCCESS); // avoids expensive de-alloc
}

int sc_main(int argc, char *argv[])
{
    try
    {
        po::options_description config("Configuration");

#ifdef DEBUG
        const int ifmap_h_default = 10;
        const int ifmap_w_default = 10;
        const int k_default = 3;
        const int c_in_default = 32;
        const int f_out_default = 32;
        const int filter_count_default = 32;
        const int channel_count_default = 18;
        const bool result_as_protobuf_default = false;
        const bool sim_bias_default = false;

        config.add_options()("help", "produce help message");
        config.add_options()("ifmap_h", po::value<int>()->default_value(ifmap_h_default),
                             "set input feature map height");
        config.add_options()("ifmap_w", po::value<int>()->default_value(ifmap_w_default),
                             "set input feature map width");
        config.add_options()("k", po::value<int>()->default_value(k_default), "set kernel size");
        config.add_options()("c_in", po::value<int>()->default_value(c_in_default), "set ifmap channel count");
        config.add_options()("f_out", po::value<int>()->default_value(f_out_default), "set weight filter count");
        config.add_options()("filter_count", po::value<int>()->default_value(filter_count_default), "set arch height");
        config.add_options()("channel_count", po::value<int>()->default_value(channel_count_default), "set arch width");
        config.add_options()("result_as_protobuf", po::bool_switch()->default_value(result_as_protobuf_default),
                             "output result as serialized protobuf to stderr");
        config.add_options()("sim_bias", po::bool_switch()->default_value(sim_bias_default),
                             "output result as serialized protobuf to stderr");

#else
        config.add_options()("help", "produce help message");
        config.add_options()("ifmap_h", po::value<int>()->required(), "set input feature map width");
        config.add_options()("ifmap_w", po::value<int>()->required(), "set input feature map height");
        config.add_options()("k", po::value<int>()->required(), "set kernel size");
        config.add_options()("c_in", po::value<int>()->required(), "set ifmap channel count");
        config.add_options()("f_out", po::value<int>()->required(), "set weight filter count");
        config.add_options()("filter_count", po::value<int>()->required(), "set arch width");
        config.add_options()("channel_count", po::value<int>()->required(), "set arch height");
        config.add_options()("result_as_protobuf", po::bool_switch()->default_value(false),
                             "output result as serialized protobuf to stderr");
        config.add_options()("sim_bias", po::bool_switch()->default_value(false), "Simulate a bias load into psum");

#endif

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, config), vm);
        po::notify(vm);

        if (vm.count("help"))
        {
            cout << config << "\n";
            return 0;
        }

        int ifmap_h = vm["ifmap_h"].as<int>();
        int ifmap_w = vm["ifmap_w"].as<int>();
        int k = vm["k"].as<int>();
        int c_in = vm["c_in"].as<int>();

        int f_out = vm["f_out"].as<int>();
        int filter_count = vm["filter_count"].as<int>();
        int channel_count = vm["channel_count"].as<int>();
        bool result_as_protobuf = vm["result_as_protobuf"].as<bool>();
        bool sim_bias = vm["sim_bias"].as<bool>();
        // bool result_as_protobuf = vm["result_as_protobuf"].as<int>();

        if (ifmap_h <= 0 || ifmap_w <= 0 || k <= 0 || c_in <= 0 || f_out <= 0 || filter_count <= 0 ||
            channel_count <= 0)
        {
            throw std::invalid_argument("all passed arguments must be positive");
        }

        if ((ifmap_h * ifmap_w) < 11)
        {
            throw std::invalid_argument("total ifmap sizes below 11 currently unsupported");
        }

        if (k != 1 && k != 3)
        {
            throw std::invalid_argument("kernel sizes not equal to 1x1 or 3x3");
        }

        cout << std::left << "Simulating arch with config:" << endl;
        cout << endl;

        cout << std::left << std::setw(20) << "filter_count" << filter_count << endl;
        cout << std::left << std::setw(20) << "channel_count" << channel_count << endl;
        cout << endl;

        cout << std::left << "With layer config:" << endl;
        cout << endl;
        cout << std::left << std::setw(20) << "ifmap_h" << ifmap_h << endl;
        cout << std::left << std::setw(20) << "ifmap_w" << ifmap_w << endl;
        cout << std::left << std::setw(20) << "k" << k << endl;
        cout << std::left << std::setw(20) << "c_in" << c_in << endl;
        cout << std::left << std::setw(20) << "f_out" << f_out << endl;

        auto operation_mode = (k == 1) ? Hero::OperationMode::RUN_1x1 : Hero::OperationMode::RUN_3x3;
        sim_and_get_results<char>(ifmap_h, ifmap_w, k, c_in, f_out, filter_count, channel_count, operation_mode,
                                  result_as_protobuf, sim_bias);
    }
    catch (std::exception &e)
    {
        cout << e.what() << "\n";
        return -1;
    }

    return 0;
}

#endif // MEM_HIERARCHY_CPP
