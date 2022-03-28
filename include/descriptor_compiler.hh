#if !defined(__DESCRIPTOR_COMPILER_CC__)
#define __DESCRIPTOR_COMPILER_CC__

#include "hero.hh"
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <fmt/format.h>
#include <vector>
#include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xarray.hpp>
#include <xtensor/xaxis_iterator.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xview.hpp>

using std::deque;
using std::vector;

namespace GenerateDescriptors
{
template <typename DataType>
void generate_and_load_arch_descriptors(Hero::Arch<DataType> &arch, int ifmap_h, int ifmap_w,
                                        xt::xarray<int> padded_weights, int ofmap_h, int ofmap_w);

} // namespace GenerateDescriptors

namespace GenerateDescriptors1x1
{
template <typename DataType> void generate_and_load_pe_program(Hero::Arch<DataType> &arch, int ifmap_h, int ifmap_w);

template <typename DataType>
void generate_and_load_psum_program(Hero::Arch<DataType> &arch, xt::xarray<int> padded_weights, int ofmap_h,
                                    int ofmap_w);

template <typename DataType>
void generate_and_load_ifmap_in_program(Hero::Arch<DataType> &arch, xt::xarray<int> padded_weights, int ifmap_h,
                                        int ifmap_w);

} // namespace GenerateDescriptors1x1

namespace GenerateDescriptors3x3
{
template <typename DataType> void generate_and_load_ssm_program(Hero::Arch<DataType> &arch, int ifmap_h, int ifmap_w);

template <typename DataType> void generate_and_load_pe_program(Hero::Arch<DataType> &arch, int ifmap_h, int ifmap_w);

template <typename DataType>
void generate_and_load_psum_program(Hero::Arch<DataType> &arch, xt::xarray<int> padded_weights, int ofmap_h,
                                    int ofmap_w);

template <typename DataType>
void generate_and_load_ifmap_in_program(Hero::Arch<DataType> &arch, xt::xarray<int> padded_weights, int ifmap_h,
                                        int ifmap_w);

template <typename DataType>
void generate_and_load_ifmap_channel_to_reuse_chain_program(Hero::Arch<DataType> &arch, xt::xarray<int> padded_weights,
                                                            int ifmap_h, int ifmap_w);

} // namespace GenerateDescriptors3x3

namespace GenerateDescriptors
{

namespace
{

template <typename DataType>
xt::xarray<int> get_ifmap_mem_run_bitmap(Hero::Arch<DataType> &arch, xt::xarray<int> padded_weights)
{

    if (arch.kmapping == Hero::KernelMapping::VERTICLE)
    {
        throw "not implemented";
    }

    // assuming horizontal kmapping

    int horizontal_kernel_size = (arch.mode == Hero::OperationMode::RUN_1x1) ? 1 : 9;
    int verticle_kernel_size = 1;
    int arch_effective_channel_count = arch.channel_count / horizontal_kernel_size;
    int arch_effective_filter_count = arch.filter_count / verticle_kernel_size;

    int output_filter_count = (padded_weights.shape()[0] / verticle_kernel_size);
    int verticle_tile_count = padded_weights.shape()[0] / arch.filter_count;

    int input_channel_count = (padded_weights.shape()[1] / horizontal_kernel_size);
    int horizontal_tile_count = input_channel_count / arch_effective_channel_count;

    cout << padded_weights << endl;

    xt::xarray<int> run_bitmap =
        xt::zeros<int>({verticle_tile_count, horizontal_tile_count, (int)arch_effective_channel_count});

    // for (int v = 0; v < verticle_tile_count; v++)
    // {
    //     for (int h = 0; h < horizontal_tile_count; h++)
    //     {
    //         for (int c = 0; c < arch_effective_channel_count; c++)
    //         {
    //             fmt::print("{}, {}, {}: {}\n", v, h, c, run_bitmap(v, h, c));
    //             fflush(stdout);
    //         }
    //     }
    // }

    for (auto filter_offset = 0; filter_offset < (output_filter_count * verticle_kernel_size);
         filter_offset += (arch_effective_filter_count * verticle_kernel_size))
    {
        for (auto channel_offset = 0; channel_offset < (input_channel_count * horizontal_kernel_size);
             channel_offset += (arch_effective_channel_count * horizontal_kernel_size))
        {
            auto tiled_view = xt::view(
                padded_weights,
                xt::range(filter_offset, filter_offset + (arch_effective_filter_count * verticle_kernel_size)),
                xt::range(channel_offset, channel_offset + (arch_effective_channel_count * horizontal_kernel_size)));

            cout << tiled_view << endl;

            int verticle_tile_idx = filter_offset / (arch_effective_filter_count * verticle_kernel_size);
            int horizontal_tile_idx = channel_offset / (arch_effective_channel_count * horizontal_kernel_size);

            for (int channel = 0; channel < arch_effective_channel_count; channel++)
            {
                if (tiled_view(0, channel * horizontal_kernel_size) != -1)
                {
                    run_bitmap(verticle_tile_idx, horizontal_tile_idx, channel) = 1;
                }
            }
        }
    }

    // for (int v = 0; v < verticle_tile_count; v++)
    // {
    //     for (int h = 0; h < horizontal_tile_count; h++)
    //     {
    //         for (int c = 0; c < arch_effective_channel_count; c++)
    //         {
    //             fmt::print("{}, {}, {}: {}\n", v, h, c, run_bitmap(v, h, c));
    //             fflush(stdout);
    //         }
    //     }
    // }

    return run_bitmap;
}

} // namespace

namespace GenerateDescriptors1x1
{
template <typename DataType> void generate_and_load_pe_program(Hero::Arch<DataType> &arch, int ifmap_h, int ifmap_w)
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
void generate_and_load_psum_program(Hero::Arch<DataType> &arch, xt::xarray<int> padded_weights, int ofmap_h,
                                    int ofmap_w)
{
    int verticle_tile_count = padded_weights.shape()[0] / arch.filter_count;
    int horizontal_tile_count = padded_weights.shape()[1] / arch.channel_count;

    int stream_size = ofmap_h * ofmap_w;

    xt::xarray<int> run_bitmap = xt::zeros<int>({verticle_tile_count, (int)arch.filter_count});
    for (auto filter_offset = 0; filter_offset < (int)padded_weights.shape()[0]; filter_offset += arch.filter_count)
    {
        for (auto channel_offset = 0; channel_offset < (int)padded_weights.shape()[1];
             channel_offset += arch.channel_count)
        {
            auto tiled_view = xt::view(padded_weights, xt::range(filter_offset, filter_offset + arch.filter_count),
                                       xt::range(channel_offset, channel_offset + arch.channel_count));
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

    // TODO: #32
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
                    program.push_back(Descriptor_2D::stream_inst(
                        v * arch.filter_count * stream_size + write_gen_idx * stream_size, stream_size - 1, 0));
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
                    program.push_back(Descriptor_2D::stream_inst((v * arch.filter_count * stream_size) +
                                                                     (read_gen_idx - arch.filter_count) * stream_size,
                                                                 stream_size - 1, 0));
                }
            }
        }
        program.push_back(Descriptor_2D::suspend_inst());
        Descriptor_2D::make_sequential(program);
        arch.psum_mem.generators.at(read_gen_idx).loadProgram(program);
    }
}

template <typename DataType>
void generate_and_load_ifmap_in_program(Hero::Arch<DataType> &arch, xt::xarray<int> padded_weights, int ifmap_h,
                                        int ifmap_w)
{
    auto run_bitmap = get_ifmap_mem_run_bitmap(arch, padded_weights);
    int verticle_tile_count = padded_weights.shape()[0] / arch.filter_count;
    int horizontal_tile_count = padded_weights.shape()[1] / arch.channel_count;
    // cout << padded_weights << endl;

    // cout << run_bitmap << endl;
    // TODO: #32 Separate loading descriptors into hero arch from compilation and provide programming api in hero
    // implementation
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
                    // TODO #42
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

} // namespace GenerateDescriptors1x1

namespace GenerateDescriptors3x3
{

template <typename DataType> void generate_and_load_ssm_program(Hero::Arch<DataType> &arch, int ifmap_h, int ifmap_w)
{
    // TODO: #41
}

template <typename DataType> void generate_and_load_pe_program(Hero::Arch<DataType> &arch, int ifmap_h, int ifmap_w)
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
void generate_and_load_psum_program(Hero::Arch<DataType> &arch, xt::xarray<int> padded_weights, int ofmap_h,
                                    int ofmap_w)
{
    int verticle_tile_count = padded_weights.shape()[0] / arch.filter_count;
    int horizontal_tile_count = padded_weights.shape()[1] / arch.channel_count;

    int stream_size = ofmap_h * ofmap_w;

    xt::xarray<int> run_bitmap = xt::zeros<int>({verticle_tile_count, (int)arch.filter_count});
    for (auto filter_offset = 0; filter_offset < (int)padded_weights.shape()[0]; filter_offset += arch.filter_count)
    {
        for (auto channel_offset = 0; channel_offset < (int)padded_weights.shape()[1];
             channel_offset += arch.channel_count)
        {
            auto tiled_view = xt::view(padded_weights, xt::range(filter_offset, filter_offset + arch.filter_count),
                                       xt::range(channel_offset, channel_offset + arch.channel_count));
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

    // TODO: #32
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
                    program.push_back(Descriptor_2D::stream_inst(
                        v * arch.filter_count * stream_size + write_gen_idx * stream_size, stream_size - 1, 0));
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
                    program.push_back(Descriptor_2D::stream_inst((v * arch.filter_count * stream_size) +
                                                                     (read_gen_idx - arch.filter_count) * stream_size,
                                                                 stream_size - 1, 0));
                }
            }
        }
        program.push_back(Descriptor_2D::suspend_inst());
        Descriptor_2D::make_sequential(program);
        arch.psum_mem.generators.at(read_gen_idx).loadProgram(program);
    }
}

template <typename DataType>
void generate_and_load_ifmap_in_program(Hero::Arch<DataType> &arch, xt::xarray<int> padded_weights, int ifmap_h,
                                        int ifmap_w)
{
    auto run_bitmap = get_ifmap_mem_run_bitmap(arch, padded_weights);
    const int verticle_tile_count = run_bitmap.shape()[0];
    const int horizontal_tile_count = run_bitmap.shape()[1];

    const int arch_effective_channel_count = arch.channel_count / 9;

    const int stream_size = ifmap_h * ifmap_w;
    const int first_two_line_size = 2 * ifmap_w;
    const int rest_of_fmap_size = (ifmap_h - 2) * ifmap_w;
    const int systolic_delay = 9;
    const int reuse_chain_delay = 6;
    const int load_delay = 1;
    const int total_stream_delay = stream_size + reuse_chain_delay + load_delay;

    for (int ag_idx = 0; ag_idx < arch_effective_channel_count; ag_idx++)
    {
        auto &ag = arch.ifmap_mem.generators.at(ag_idx);

        std::deque<Descriptor_2D> program;

        unsigned int channel_start_delay = (ag_idx * systolic_delay);
        channel_start_delay = (channel_start_delay == 0) ? 0 : (channel_start_delay - 1);

        auto systolic_delay_inst = Descriptor_2D::delay_inst(channel_start_delay);
        program.push_back(systolic_delay_inst);

        for (int v = 0; v < verticle_tile_count; v++)
        {
            for (int h = 0; h < horizontal_tile_count; h++)
            {
                int active = run_bitmap(v, h, ag_idx);
                if (active)
                {
                    int stream_start_idx = h * arch_effective_channel_count * stream_size + ag_idx * stream_size;
                    auto stream_first_two_lines_inst =
                        Descriptor_2D::stream_inst(stream_start_idx, first_two_line_size - 1, 0);
                    program.push_back(stream_first_two_lines_inst);

                    auto reuse_chain_delay_inst = Descriptor_2D::delay_inst(reuse_chain_delay - 1 - 2 * load_delay);
                    program.push_back(reuse_chain_delay_inst);

                    int rest_of_fmap_start = stream_start_idx + first_two_line_size;
                    auto stream_rest_of_ifmap_inst =
                        Descriptor_2D::stream_inst(rest_of_fmap_start, rest_of_fmap_size - 1, 0);
                    program.push_back(stream_rest_of_ifmap_inst);
                }
                else
                {
                    auto delay_inst = Descriptor_2D::delay_inst(total_stream_delay - 1);
                    program.push_back(delay_inst);
                }
                const int channel_memories_to_wait_for = (arch.channel_count - 1);
                auto delay_inst = Descriptor_2D::delay_inst((channel_memories_to_wait_for * total_stream_delay) - 1);
                program.push_back(delay_inst);
            }
        }
        program.push_back(Descriptor_2D::suspend_inst());
        Descriptor_2D::make_sequential(program);
        ag.loadProgram(program);
    }
}

template <typename DataType>
void generate_and_load_ifmap_channel_to_reuse_chain_program(Hero::Arch<DataType> &arch, xt::xarray<int> padded_weights,
                                                            int ifmap_h, int ifmap_w)
{
    throw "Not implemented";
}

} // namespace GenerateDescriptors3x3

template <typename DataType>
void generate_and_load_arch_descriptors(Hero::Arch<DataType> &arch, int ifmap_h, int ifmap_w,
                                        xt::xarray<int> padded_weights, int ofmap_h, int ofmap_w)
{

    switch (arch.mode)
    {
    case Hero::OperationMode::RUN_1x1:
        GenerateDescriptors1x1::generate_and_load_pe_program(arch, ifmap_h, ifmap_w);
        GenerateDescriptors1x1::generate_and_load_ifmap_in_program(arch, padded_weights, ifmap_h, ifmap_w);
        GenerateDescriptors1x1::generate_and_load_psum_program(arch, padded_weights, ofmap_h, ofmap_w);
        break;
    case Hero::OperationMode::RUN_3x3:
        GenerateDescriptors3x3::generate_and_load_ifmap_in_program(arch, padded_weights, ifmap_h, ifmap_w);
        GenerateDescriptors3x3::generate_and_load_ssm_program(arch, ifmap_h, ifmap_w);
        GenerateDescriptors3x3::generate_and_load_pe_program(arch, ifmap_h, ifmap_w);
        GenerateDescriptors3x3::generate_and_load_ifmap_channel_to_reuse_chain_program(arch, padded_weights, ifmap_h,
                                                                                       ifmap_w);
        GenerateDescriptors3x3::generate_and_load_psum_program(arch, padded_weights, ofmap_h, ofmap_w);
        break;
    default:
        throw std::invalid_argument("Invalid architecture operation mode");
        break;
    }
}

} // namespace GenerateDescriptors

#endif